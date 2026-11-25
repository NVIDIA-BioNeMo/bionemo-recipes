# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate from prepared encoder features and a trained Stitch decoder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="GRPO config defining the prepared inputs and vLLM plugin")
    parser.add_argument("--model", help="Consolidated decoder directory (defaults to the GRPO config's SFT donor)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--record-index", type=int, default=0)
    args = parser.parse_args()

    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf
    from vllm import LLM, SamplingParams

    from nemotron_stitch.nemo_rl.transport import resolve_callback

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    policy = config["policy"]
    train = config["data"]["train"].copy()
    dataset_cls = resolve_callback(train.pop("dataset_name"))
    dataset = dataset_cls(**train)
    dataset.set_task_spec(config["data"]["default"])
    dataset.set_processor()

    tokenizer_config = policy["tokenizer"]
    tokenizer = get_tokenizer(tokenizer_config)
    row = dataset.processor(
        dataset.dataset[args.record_index],
        dataset.task_spec,
        tokenizer,
        policy["max_total_sequence_length"],
        args.record_index,
    )

    generation = policy["generation"]
    resolve_callback(generation["mm_plugin_callback"])()
    overrides = {**policy["hf_config_overrides"], "architectures": [generation["mm_architecture"]]}
    llm = LLM(
        model=args.model or policy["model_name"],
        tokenizer=tokenizer_config["name"],
        trust_remote_code=True,
        hf_overrides=overrides,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=policy["precision"],
        max_model_len=generation["vllm_cfg"]["max_model_len"],
        enforce_eager=generation["vllm_cfg"]["enforce_eager"],
        **generation["vllm_kwargs"],
    )
    request = {
        "prompt_token_ids": tokenizer.encode(row["vllm_content"], add_special_tokens=False),
        "multi_modal_data": row["vllm_multi_modal_data"],
    }
    result = llm.generate(request, SamplingParams(temperature=0, max_tokens=128))
    print(result[0].outputs[0].text)


if __name__ == "__main__":
    main()
