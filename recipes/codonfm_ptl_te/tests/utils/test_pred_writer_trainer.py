# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Cuma Can Bilmez. All rights reserved.
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

"""Real evaluate/Trainer regression using four rows and two external CPU ranks.

The fixture is deliberately independent of Encodon/model weights. Observation
does not call native hooks, alter samplers, or add synchronization operations.
"""

import argparse
import datetime
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
import torch.distributed as dist
from lightning.fabric.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset

from src import tasks
from src.data.datamodule import CodonFMDataModule
from src.data.token_packing_batch_sampler import TokenPackingBatchSampler
from src.utils.pred_writer import PredWriter


_RECORD = {}
_EVENT_HANDLES = {}
_MODEL_ROWS = ((0, 10), (1, 11), (2, 12), (3, 13))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _event(name, **fields):
    item = {"event": name, "pid": os.getpid(), "time_ns": time.monotonic_ns(), **fields}
    _RECORD["events"].append(item)
    _EVENT_HANDLES["stream"].write(json.dumps(item, allow_nan=False) + "\n")
    _EVENT_HANDLES["stream"].flush()


def _source(obj):
    path = inspect.getsourcefile(obj)
    return {"path": path, "sha256": _sha(path), "module": obj.__module__, "name": obj.__qualname__}


def _identity(item):
    return item


class _Rows(Dataset):
    def __init__(self, lengths, process_item):
        self.lengths, self.process_item = lengths, process_item

    def __len__(self):
        return len(self.lengths)

    def get_sequence_length(self, index):
        return self.lengths[index]

    def __getitem__(self, index):
        return self.process_item(
            {"ids": torch.tensor(index), "features": torch.tensor(_MODEL_ROWS[index], dtype=torch.float32)}
        )


class _DatasetFactory:
    def __init__(self, lengths, seed):
        self.lengths = lengths
        _event("dataset_factory", seed=seed, lengths=list(lengths))

    def get_test(self, process_item):
        _event("get_test", process_item=process_item.__qualname__)
        return _Rows(self.lengths, process_item)


class _PredictionModel(pl.LightningModule):
    def configure_model(self):
        created = not hasattr(self, "anchor")
        if created:
            self.anchor = torch.nn.Parameter(torch.zeros(1))
        _event("configure_model", created=created, caller=inspect.stack()[1].function)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids = batch["ids"].tolist()
        _RECORD["observed_batches"].append(ids)
        _event("predict_step", batch_idx=batch_idx, ids=ids)
        return {"ids": batch["ids"], "embeddings": batch["features"] + self.anchor * 0}


def _sampler_chain(loader):
    chain = [{"role": "dataloader", **_source(type(loader))}]
    current, seen = loader.batch_sampler, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        fields = {
            key: getattr(current, key)
            for key in ("shuffle", "drop_last", "rank", "num_replicas", "samples_yielded")
            if hasattr(current, key)
        }
        chain.append({"role": "batch_sampler_or_sampler", **_source(type(current)), "fields": fields})
        if hasattr(current, "_batch_sampler"):
            current = current._batch_sampler
        elif hasattr(current, "sampler"):
            current = current.sampler
        else:
            break
    return chain


class _BeforeWriter(pl.Callback):
    def on_predict_start(self, trainer, pl_module):
        loader = trainer.predict_dataloaders
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        self.loader = loader
        _RECORD["sampler_chain_start"] = _sampler_chain(loader)
        _RECORD["actual_runtime"] = {
            "trainer": _source(type(trainer)),
            "strategy": _source(type(trainer.strategy)),
            "cluster_environment": _source(type(trainer.strategy.cluster_environment)),
            "datamodule": _source(type(trainer.datamodule)),
            "model_module": type(pl_module).__module__,
            "model_is_lightning_pytorch": isinstance(pl_module, pl.LightningModule),
            "model_base_namespace": pl.LightningModule.__module__,
            "callback_modules": [type(x).__module__ + "." + type(x).__qualname__ for x in trainer.callbacks],
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "backend": dist.get_backend(),
            "device": str(pl_module.device),
            "limit_predict_batches": trainer.limit_predict_batches,
            "fast_dev_run": trainer.fast_dev_run,
            "use_distributed_sampler": trainer._accelerator_connector.use_distributed_sampler,
            "native_data_settings": {
                key: getattr(trainer.datamodule, key)
                for key in (
                    "train_batch_size",
                    "val_batch_size",
                    "max_tokens_per_batch",
                    "is_evaluation",
                    "world_size",
                    "num_workers",
                    "persistent_workers",
                    "shuffle",
                    "global_batch_size",
                    "micro_batch_size",
                )
            },
        }
        _event("on_predict_start", rank=trainer.global_rank)

    def on_predict_epoch_end(self, trainer, pl_module):
        _RECORD["sampler_chain_end"] = _sampler_chain(self.loader)
        directory = Path(_RECORD["phase_dir"])
        evidence = directory / "shard_evidence" / f"rank{trainer.global_rank}"
        evidence.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for path in sorted((directory / "shards").glob(f"*_rank_{trainer.global_rank}_batch_*.npy")):
            shutil.copyfile(path, evidence / path.name)
            hashes[path.name] = _sha(evidence / path.name)
        _RECORD["shard_evidence_hashes"] = hashes
        _event("before_writer_epoch_end", rank=trainer.global_rank, shard_count=len(hashes))


class _AfterWriter(pl.Callback):
    def on_predict_epoch_end(self, trainer, pl_module):
        _event("after_writer_epoch_end", rank=trainer.global_rank)

    def on_predict_end(self, trainer, pl_module):
        _event("on_predict_end", rank=trainer.global_rank)


def _install_readonly_trace():
    targets = {
        tasks.evaluate.__code__: "tasks.evaluate",
        pl.Trainer.predict.__code__: "Trainer.predict",
        CodonFMDataModule.setup.__code__: "CodonFMDataModule.setup",
        CodonFMDataModule.predict_dataloader.__code__: "CodonFMDataModule.predict_dataloader",
        CodonFMDataModule._make_token_packed_eval_dataloader.__code__: "CodonFMDataModule._make_token_packed_eval_dataloader",
        PredWriter.write_on_batch_end.__code__: "PredWriter.write_on_batch_end",
        PredWriter._save_predictions.__code__: "PredWriter._save_predictions",
        PredWriter.on_predict_epoch_end.__code__: "PredWriter.on_predict_epoch_end",
        PredWriter._merge_predictions.__code__: "PredWriter._merge_predictions",
    }

    def local(frame, event, arg):
        if event in ("return", "exception"):
            fields = {"function": targets[frame.f_code], "path": frame.f_code.co_filename, "line": frame.f_lineno}
            if event == "exception":
                fields.update(error_type=arg[0].__name__, error_message=str(arg[1]))
            _event("native_" + event, **fields)
        return local

    def trace(frame, event, arg):
        if event == "call" and frame.f_code in targets:
            _event("native_call", function=targets[frame.f_code], path=frame.f_code.co_filename, line=frame.f_lineno)
            return local
        return None

    sys.settrace(trace)


def _worker(phase, case, lengths):
    rank = int(os.environ["RANK"])
    rank_dir = phase / f"rank{rank}"
    _EVENT_HANDLES["stream"] = (rank_dir / "events.jsonl").open("x")
    _RECORD.update(
        {
            "rank": rank,
            "pid": os.getpid(),
            "phase_dir": str(phase),
            "case": case,
            "lengths": lengths,
            "events": [],
            "observed_batches": [],
            "status": "STARTED",
        }
    )
    code = 1
    try:
        _RECORD["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "lightning",
                "pytorch-lightning",
                "torchmetrics",
                "huggingface-hub",
                "safetensors",
            )
        }
        _RECORD["torch_cuda_version"] = torch.version.cuda
        _RECORD["sources"] = {
            "evaluate": _source(tasks.evaluate),
            "trainer": _source(pl.Trainer),
            "datamodule": _source(CodonFMDataModule),
            "token_sampler": _source(TokenPackingBatchSampler),
            "writer": _source(PredWriter),
            "test_fixture": {"path": __file__, "sha256": _sha(__file__)},
        }
        data_kwargs = dict(
            dataset=functools.partial(_DatasetFactory, lengths=lengths),
            seed=123,
            world_size=2,
            train_batch_size=None,
            val_batch_size=None,
            max_tokens_per_batch=4,
            is_evaluation=True,
            num_workers=0,
            persistent_workers=False,
            shuffle=False,
            pin_memory=False,
            process_item=_identity,
        )
        writer_kwargs = dict(
            output_dir=str(phase / "shards"),
            write_interval="batch",
            caching_interval=1,
            merge_on_epoch_end=True,
            delete_after_merge=True,
        )
        trainer_kwargs = dict(
            accelerator="cpu",
            devices=2,
            num_nodes=1,
            strategy=DDPStrategy(
                process_group_backend="gloo",
                cluster_environment=TorchElasticEnvironment(),
                timeout=datetime.timedelta(seconds=45),
            ),
            precision="32-true",
            use_distributed_sampler=False,
            limit_predict_batches=1.0,
            fast_dev_run=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(phase / "trainer"),
        )
        _RECORD["configuration"] = {
            "data": {k: v for k, v in data_kwargs.items() if k not in ("dataset", "process_item")},
            "writer": writer_kwargs,
            "trainer": {k: v for k, v in trainer_kwargs.items() if k != "strategy"},
            "strategy": "DDPStrategy(gloo, TorchElasticEnvironment, timeout=45s)",
            "launcher": "two subprocesses with real external-process Lightning environment; not torchrun",
        }
        model, data = _PredictionModel(), CodonFMDataModule(**data_kwargs)
        config = {
            "log": False,
            "model": model,
            "data": data,
            "trainer": trainer_kwargs,
            "callbacks": {"before": _BeforeWriter(), "writer": PredWriter(**writer_kwargs), "after": _AfterWriter()},
        }
        checkpoint = str(phase / "deliberately_missing.ckpt")
        assert not Path(checkpoint).exists()
        _RECORD["checkpoint_path"] = checkpoint
        _RECORD["checkpoint_exists_before"] = False
        _install_readonly_trace()
        _event("evaluate_invocation")
        tasks.evaluate(config=config, config_dict={}, model_ckpt_path=checkpoint, out_dir=str(phase / "task_output"))
        _event("evaluate_returned")
        _RECORD["status"] = "SUCCESS"
        code = 0
    except BaseException as exc:
        _RECORD.update(
            status="ERROR", error_type=type(exc).__name__, error_message=str(exc), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        sys.settrace(None)
        _RECORD["exit_code_intent"] = code
        _RECORD["dist_initialized_at_worker_finish"] = dist.is_initialized()
        if rank == 0:
            files = sorted((phase / "shards").glob("*_merged.npy"))
            _RECORD["output_file_hashes"] = {p.name: _sha(p) for p in files}
            _RECORD["remaining_native_shards"] = sorted(
                p.name for p in (phase / "shards").glob("*_rank_*_batch_*.npy")
            )
        _write(rank_dir / "record.json", _RECORD)
        _EVENT_HANDLES["stream"].close()
    return code


def _launch_trial(phase, case, lengths):
    phase.mkdir(parents=True, exist_ok=False)
    recipe_root = Path(__file__).absolute().parents[2]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    children = []
    deadline = time.monotonic() + 120
    for rank in (0, 1):
        rank_dir = phase / f"rank{rank}"
        rank_dir.mkdir()
        env = os.environ.copy()
        env.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_WORLD_SIZE="2",
            GROUP_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            GLOO_SOCKET_IFNAME="lo",
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            MPLCONFIGDIR=str(rank_dir / "mplconfig"),
            PYTHONPATH=str(recipe_root),
        )
        command = [
            sys.executable,
            str(Path(__file__).absolute()),
            "--worker",
            "--phase",
            str(phase),
            "--case",
            case,
            "--lengths",
            json.dumps(lengths),
        ]
        stdout_path, stderr_path = rank_dir / "process.stdout.txt", rank_dir / "process.stderr.txt"
        stdout, stderr = stdout_path.open("x"), stderr_path.open("x")
        process = subprocess.Popen(command, cwd=recipe_root, env=env, stdout=stdout, stderr=stderr)
        observed = {
            "rank": rank,
            "pid": process.pid,
            "command": command,
            "returncode": None,
            "timed_out": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        children.append((process, stdout, stderr, observed))
    for process, stdout, stderr, observed in children:
        try:
            observed["returncode"] = process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            observed["timed_out"] = True
            process.kill()
            observed["returncode"] = process.wait()
        finally:
            stdout.close()
            stderr.close()
        _write(phase / f"rank{observed['rank']}" / "process.command.json", observed)
    result = {
        "pid": os.getpid(),
        "case": case,
        "lengths": lengths,
        "recipe_root": str(recipe_root),
        "master_address": "127.0.0.1",
        "master_port": port,
        "children": [x[3] for x in children],
        "launcher_type": "subprocess parent + native TorchElasticEnvironment",
    }
    _write(phase / "phase.json", result)
    return result


@pytest.mark.integration
@pytest.mark.parametrize(
    "case,lengths", [("equal", [2, 2, 2, 2]), ("unequal", [2, 3, 2, 3])], ids=["equal", "unequal"]
)
def test_predict_writer_preserves_token_packed_order(tmp_path, case, lengths):
    phase = Path(os.environ.get("CODONFM_PRED_WRITER_TRIAL_DIR", str(tmp_path / "trial"))).absolute()
    result = _launch_trial(phase, case, lengths)
    failures = [x for x in result["children"] if x["returncode"] != 0 or x["timed_out"]]
    details = "\n".join(Path(x["stderr_path"]).read_text() for x in failures)
    assert not failures, details
    ids = np.load(phase / "shards/ids_merged.npy", allow_pickle=False)
    embeddings = np.load(phase / "shards/embeddings_merged.npy", allow_pickle=False)
    assert ids.tolist() == [0, 1, 2, 3]
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(embeddings, [[0, 10], [1, 11], [2, 12], [3, 13]])
    assert not list((phase / "shards").glob("*_rank_*_batch_*.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--case", choices=("equal", "unequal"), required=True)
    parser.add_argument("--lengths", required=True)
    args = parser.parse_args()
    raise SystemExit(_worker(args.phase, args.case, json.loads(args.lengths)))
