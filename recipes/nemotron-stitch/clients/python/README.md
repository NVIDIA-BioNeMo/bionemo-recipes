# BioNeMo Perception Container Python client

`bionemo-perception-client` is the **container client** included with this
Nemotron Stitch recipe. It queries an encoder service over HTTP; it does not
include or launch an encoder model or server. **Container publishing is still
work in progress (WIP)**. The container implementation is private. To use this
client, obtain the URL of an independently deployed compatible service.

The package provides typed Evo 2, ESM-3, and KERMT clients and a model-neutral
parser for `BNMOFWD1` tensor envelopes (and legacy `BNMOFWD2` framing). It sends canonical `/v1/forward` requests,
requests `application/vnd.bnmo.tensor+binary; version=1`, and validates payload
geometry and the XXH64 checksum. BF16 and FP32 wire tensors are supported.

## Install from this recipe

From `recipes/nemotron-stitch`:

```bash
python -m pip install './clients/python[torch]'
```

For NumPy feature preparation, install `./clients/python[numpy]` instead.
For transport and parsing alone, omit the extra. No access to the
private container repository is required. The shared examples Dockerfile installs the
client in its driver and worker environments.

## Query a container

Set `PERCEPTION_URL` to your deployed service's origin or full `/v1/forward`
URL. Set `PERCEPTION_LAYER` to a capture layer exposed by that service. Output
layers and their feature widths depend on the deployed model.

```python
import os

from bionemo_perception import Evo2Client

client = Evo2Client(
    os.environ["PERCEPTION_URL"],
    timeout=900,
    token=os.environ.get("PERCEPTION_TOKEN"),
)
response = client.forward(
    sequence="ACGT",
    outputs=[
        {
            "layer": os.environ["PERCEPTION_LAYER"],
            "name": "features",
            "pooling": {"kind": "none"},
        }
    ],
)
for tensor in response.tensors:
    print(tensor.name, tensor.shape, tensor.dtype)
```

`ESM3Client.forward` accepts `sequence`, `outputs`, and optional tracks such as
`structure_coords` and `per_res_plddt`. `KERMTClient.forward` accepts `smiles`
and `outputs` instead. Both use the same transport and response parser.
Custom clients can use `PerceptionClient[RequestT].forward_payload(request)`.
The request schemas are exported from `bionemo_perception` for type checking.

`token=` sends `Authorization: Bearer <token>`. Surrounding whitespace is
stripped; empty tokens and tokens containing internal whitespace are rejected.
Bearer authorization is sent only to the initial URL and is omitted on all
redirects, including redirects to another origin or from HTTPS to HTTP. Use the
service's final forward URL for authenticated requests.
Omit the token for an open service. Non-successful HTTP responses raise
`PerceptionRequestError`, which exposes `status_code`, `body`, and `url`.
Transport errors and timeouts propagate from Python's `urllib`.

## Use response tensors in Stitch

The application selects the required response tensor and converts its geometry
to the projector's feature contract. For example, after the single-output
request above:

```python
wire = response.tensors[0]
if wire.dtype == "bfloat16-le":
    features = wire.torch_bfloat16().clone()
else:
    features = wire.torch_float32().clone()
```

For nonempty payloads, these conversion methods return views over the response
buffer; clone before modifying the tensor. Zero-extent outputs return empty
CPU tensors with the declared shape and dtype. In your application, validate the selected
output, feature width, token count, and any batch or ragged row boundaries
before passing features to the projector. Do not assume every model emits
`[T, F]` directly. Prompt placeholder counts must match the projector's output
token count. For Stitch's NumPy cache helper, `wire.numpy_float32()` widens BF16 exactly
and returns a view for FP32 without requiring PyTorch; a custom data-loading boundary can keep PyTorch
features directly, including BF16.

For an existing response body, use `parse_response(body_bytes)` or its alias
`parse_forward_response`. `response.tensor(name)` retrieves a unique named
tensor; `response.tensor_metadata` includes output names, resolved layers,
pooling, and any row offsets.

The local CLIP LLaVA example continues to use its own encoder. See the
[adaptation guide](../../examples/llava/quickstart.md#container-client-optional)
for the application boundary used by other encoders.

## Tests

From `clients/python`:

```bash
python -m pip install '.[test]'
python -m pytest tests
python -m mypy src tests/typecheck_client.py
```

Tests use synthetic envelopes and mocked HTTP transport; they require no
running encoder container. Live service compatibility requires separate
qualification against the deployed model and container version.

## Source snapshot

Imported from the Python client snapshot at commit
`aa831c731f30e9f7bb878e9e29c0e61848f36e1d`, with recipe packaging, documentation,
formatting, and tests maintained here. The client is Apache-2.0 licensed; see
[LICENSE](LICENSE). Container implementation files are not included.
