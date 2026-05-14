"""
nim-cpu-bridge: converter/ngc_to_gguf.py

Downloads SafeTensors weights from NGC (or a local NIM cache directory)
and converts them to GGUF format for llama.cpp / ctransformers.

On first run:
  1. Reads the NGC model manifest from .cache/nim/<model>/
  2. Identifies SafeTensors shards
  3. Invokes llama.cpp's convert_hf_to_gguf.py
  4. Quantises to the requested level (default Q4_K_M)
  5. Caches the result alongside the source weights

Subsequent runs detect the cached GGUF and skip conversion.

Apache 2.0 — see LICENSE
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("ncb.converter")


# ── Quantisation levels supported by llama.cpp ────────────────────────────────
VALID_QUANT_TYPES = frozenset({
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
    "F16", "F32",
})


class ConversionError(RuntimeError):
    pass


# ── NGC weight discovery ──────────────────────────────────────────────────────

def find_nim_cache_dir(model_tag: str) -> Path | None:
    """
    NIM caches downloaded weights at ~/.cache/nim/<model_slug>/<version>/.
    Given a model tag like 'meta/llama-3.1-8b-instruct', try to locate it.
    """
    slug = model_tag.replace("/", "--").replace(":", "--")
    candidates = [
        Path(os.getenv("NIM_CACHE_PATH", str(Path.home() / ".cache" / "nim"))),
        Path("/opt/nim/.cache"),
        Path("/tmp/nim_cache"),
    ]
    for base in candidates:
        for candidate in base.glob(f"**/{slug}*"):
            if candidate.is_dir():
                logger.info("Found NIM cache at %s", candidate)
                return candidate
        # Also try the raw base (compose mounts)
        if (base / slug).exists():
            return base / slug
    return None


def find_safetensors_shards(cache_dir: Path) -> list[Path]:
    """Locate all .safetensors shards under the cache directory."""
    shards = sorted(cache_dir.glob("**/*.safetensors"))
    if not shards:
        raise ConversionError(f"No .safetensors files found under {cache_dir}")
    logger.info("Found %d SafeTensors shards", len(shards))
    return shards


def find_model_index(cache_dir: Path) -> Path | None:
    """Look for model.safetensors.index.json (sharded) or config.json."""
    for name in ("model.safetensors.index.json", "config.json"):
        p = next(cache_dir.glob(f"**/{name}"), None)
        if p:
            return p.parent
    return None


# ── llama.cpp converter wrapper ───────────────────────────────────────────────

def _find_llama_cpp_convert_script() -> Path:
    """Locate convert_hf_to_gguf.py from the installed llama-cpp-python package."""
    try:
        import llama_cpp  # noqa: F401
        pkg_dir = Path(llama_cpp.__file__).parent
        script = pkg_dir / "llama.cpp" / "convert_hf_to_gguf.py"
        if script.exists():
            return script
    except ImportError:
        pass

    # Fallback: check PATH / well-known install locations
    for search in [Path("/usr/local/lib"), Path("/opt")]:
        matches = list(search.glob("**/convert_hf_to_gguf.py"))
        if matches:
            return matches[0]

    raise ConversionError(
        "convert_hf_to_gguf.py not found. "
        "Install llama-cpp-python with: pip install llama-cpp-python"
    )


def convert_safetensors_to_gguf(
    model_dir: Path,
    output_path: Path,
    quant_type: str = "Q4_K_M",
) -> Path:
    """
    Run llama.cpp's HuggingFace-to-GGUF converter, then quantise.
    Returns the path to the final quantised GGUF.
    """
    if quant_type not in VALID_QUANT_TYPES:
        raise ConversionError(
            f"Invalid quantisation type '{quant_type}'. "
            f"Choose from: {sorted(VALID_QUANT_TYPES)}"
        )

    convert_script = _find_llama_cpp_convert_script()
    fp16_path = output_path.with_suffix(".f16.gguf")
    quantised_path = output_path.with_name(
        output_path.stem + f".{quant_type}.gguf"
    )

    if quantised_path.exists():
        logger.info("Cached GGUF found at %s — skipping conversion", quantised_path)
        return quantised_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert HF SafeTensors → F16 GGUF
    logger.info("Converting SafeTensors → F16 GGUF …")
    cmd_convert = [
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(fp16_path),
        "--outtype", "f16",
    ]
    logger.debug("Running: %s", " ".join(cmd_convert))
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(
            f"convert_hf_to_gguf.py failed:\n{result.stderr}"
        )
    logger.info("F16 GGUF written to %s", fp16_path)

    # Step 2: Quantise F16 → target quant
    logger.info("Quantising %s → %s …", fp16_path.name, quant_type)
    quantise_bin = _find_llama_quantise_bin()
    if quantise_bin:
        cmd_quant = [
            str(quantise_bin),
            str(fp16_path),
            str(quantised_path),
            quant_type,
        ]
        result = subprocess.run(cmd_quant, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Quantisation failed, using F16 GGUF: %s", result.stderr)
            shutil.copy(fp16_path, quantised_path)
        else:
            fp16_path.unlink(missing_ok=True)  # clean up intermediate
    else:
        logger.warning("llama-quantize not found — using F16 GGUF (larger, slower)")
        shutil.copy(fp16_path, quantised_path)
        fp16_path.unlink(missing_ok=True)

    logger.info("Model ready at %s (%.1f GB)",
                quantised_path,
                quantised_path.stat().st_size / (1024**3))
    return quantised_path


def _find_llama_quantise_bin() -> Path | None:
    """Find llama-quantize binary."""
    for name in ("llama-quantize", "quantize"):
        path = shutil.which(name)
        if path:
            return Path(path)
    # Check llama_cpp package bin directory
    try:
        import llama_cpp
        pkg_bin = Path(llama_cpp.__file__).parent / "llama-quantize"
        if pkg_bin.exists():
            return pkg_bin
    except ImportError:
        pass
    return None


# ── ONNX export path ──────────────────────────────────────────────────────────

def convert_safetensors_to_onnx(
    model_dir: Path,
    output_dir: Path,
    opset: int = 17,
) -> Path:
    """
    Export HuggingFace model to ONNX using optimum.
    Used when NCB_BACKEND=onnxruntime is requested.
    """
    onnx_model_path = output_dir / "model.onnx"
    if onnx_model_path.exists():
        logger.info("Cached ONNX model found — skipping export")
        return onnx_model_path

    try:
        from optimum.exporters.onnx import main_export  # type: ignore
    except ImportError:
        raise ConversionError(
            "optimum not installed. Run: pip install optimum[onnxruntime]"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to ONNX (opset %d) …", opset)
    main_export(
        model_name_or_path=str(model_dir),
        output=output_dir,
        task="text-generation-with-past",
        opset=opset,
        device="cpu",
    )
    logger.info("ONNX model exported to %s", output_dir)
    return onnx_model_path


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_model(
    model_tag: str,
    cache_base: str | None = None,
    quant_type: str = "Q4_K_M",
    backend: str = "llama_cpp",
) -> Path:
    """
    Main entry: given a NIM model tag, locate weights and ensure a
    CPU-ready model file exists. Returns path to the model file.

    Args:
        model_tag:   e.g. "meta/llama-3.1-8b-instruct"
        cache_base:  override cache root; defaults to $NIM_CACHE_PATH
        quant_type:  GGUF quantisation level
        backend:     "llama_cpp" or "onnxruntime"
    """
    nim_dir = find_nim_cache_dir(model_tag)
    if not nim_dir:
        raise ConversionError(
            f"NIM cache not found for model '{model_tag}'. "
            "Ensure the NIM container has downloaded weights, or set NIM_CACHE_PATH."
        )

    model_dir = find_model_index(nim_dir)
    if not model_dir:
        raise ConversionError(f"Could not locate model config/index under {nim_dir}")

    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_tag)
    out_base = Path(cache_base or nim_dir) / "ncb_converted" / slug

    if backend == "onnxruntime":
        return convert_safetensors_to_onnx(model_dir, out_base / "onnx")
    else:
        return convert_safetensors_to_gguf(
            model_dir, out_base / f"{slug}.gguf", quant_type
        )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="nim-cpu-bridge: NGC → GGUF converter")
    parser.add_argument("model_tag",  help='e.g. "meta/llama-3.1-8b-instruct"')
    parser.add_argument("--quant",    default="Q4_K_M", help="GGUF quant type")
    parser.add_argument("--backend",  default="llama_cpp", choices=["llama_cpp", "onnxruntime"])
    parser.add_argument("--cache",    default=None, help="Override cache root")
    args = parser.parse_args()

    out = prepare_model(args.model_tag, args.cache, args.quant, args.backend)
    print(f"Model ready: {out}")
