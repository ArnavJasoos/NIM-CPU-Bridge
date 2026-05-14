# Contributing to nim-cpu-bridge

Thank you for your interest in contributing! This document covers the process
for submitting bug reports, feature requests, and pull requests.

---

## Before You Start

- Check the [open issues](../../issues) to avoid duplicate work.
- For large features, open an issue first to discuss the design.
- All contributions must be compatible with the **Apache 2.0** license.

---

## Development Setup

```bash
git clone https://github.com/your-org/nim-cpu-bridge
cd nim-cpu-bridge

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install pytest httpx pytest-asyncio   # test extras
```

Build the shim locally (Linux only, requires gcc):

```bash
cd src/shim && make && cd ../..
```

---

## Running Tests

Tests run without a real GPU, NIM image, or model weights — all hardware
is mocked by `tests/conftest.py`.

```bash
pytest tests/ -v
```

To run a specific test:

```bash
pytest tests/test_integration.py::TestBackendSelection -v
```

---

## Pull Request Guidelines

1. **One concern per PR** — keep changes focused and reviewable.
2. **Add a test** for any new CUDA symbol stub, backend path, or converter logic.
3. **Keep the shim in C** — no C++ to minimise binary size and ABI risk.
4. **Do not commit binaries** — `.so` files, `.gguf`, `.safetensors` are gitignored.
5. **Pass all tests** — run `pytest tests/ -v` against a tiny model before submitting.
6. **Update docs** — if behaviour changes, update `docs/deep-dive.md` and the relevant
   section of `README.md`.

---

## CUDA Symbol Stubs

When adding a new CUDA symbol to `src/shim/cuda_shim.c`:

- Return `CUDA_SUCCESS` (0) unless the symbol genuinely needs to return data.
- For symbols that return device properties, use the `fake_gpu_name()` /
  `fake_vram_bytes()` helpers so values remain env-var configurable.
- Add the symbol to the coverage table in `docs/deep-dive.md`.
- Add a test that calls the symbol via `ctypes` and asserts the return value.

---

## Code Style

- **C (shim):** follow the style in `cuda_shim.c` — C11, `gcc -Wall -Wextra`.
- **Python:** PEP 8; type annotations required for all public functions.
- **YAML:** 2-space indent; comment every non-obvious field.

---

## Reporting Issues

Please include:

- NIM image tag and version
- Host OS and CPU model
- `NCB_LOG_LEVEL=DEBUG` output from the sidecar and NIM containers
- The exact error message or unexpected behaviour
