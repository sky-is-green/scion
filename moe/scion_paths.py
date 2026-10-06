"""Path defaults for the MoE harness. Every default is environment-overridable.

SCION_WORKSPACE          workspace root (default: the parent directory of this repo)
SCION_MODELS             model directory (default: $SCION_WORKSPACE/models)
SCION_HIVEBENCH          sibling hivebench checkout (default: $SCION_WORKSPACE/hivebench)
SCION_HIVEBENCH_ARTIFACTS hivebench artifacts directory (default: $SCION_HIVEBENCH/artifacts)
SCION_STORAGE            scratch/storage directory (default: $SCION_WORKSPACE/storage)
LLAMA_BIN                llama.cpp build bin directory (default: ~/llama.cpp/build/bin)
GGUF_PY                  llama.cpp gguf-py directory (default: ~/llama.cpp/gguf-py)
QWEN4EXP_GGUF_PY         qwen4exp fork gguf-py (default: $SCION_WORKSPACE/llama-qwen4exp/gguf-py)
SCION_QWEN4EXP_BIN       qwen4exp fork build bin (default: $SCION_WORKSPACE/llama-qwen4exp/build-q4exp-proto/bin)
AUTOGRID_REPO            autogrid fork checkout (default: $SCION_WORKSPACE/autogrid)
SCION_Q4_MODEL_DIR       qwen4exp FP8 model directory (default: $SCION_MODELS/qwen38-flashnext-fp8)
SCION_RELEASE_GGUF       release GGUF path (default: $SCION_WORKSPACE/qwen35-release.gguf)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("SCION_WORKSPACE", REPO.parent))
MODELS = Path(os.environ.get("SCION_MODELS", WORKSPACE / "models"))
HIVEBENCH = Path(os.environ.get("SCION_HIVEBENCH", WORKSPACE / "hivebench"))
HIVE_ARTIFACTS = Path(os.environ.get("SCION_HIVEBENCH_ARTIFACTS", HIVEBENCH / "artifacts"))
STORAGE = Path(os.environ.get("SCION_STORAGE", WORKSPACE / "storage"))
LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", Path.home() / "llama.cpp" / "build" / "bin"))
GGUF_PY = Path(os.environ.get("GGUF_PY", Path.home() / "llama.cpp" / "gguf-py"))
QWEN4EXP_GGUF_PY = Path(os.environ.get("QWEN4EXP_GGUF_PY", WORKSPACE / "llama-qwen4exp" / "gguf-py"))
QWEN4EXP_BIN = Path(os.environ.get("SCION_QWEN4EXP_BIN", WORKSPACE / "llama-qwen4exp" / "build-q4exp-proto" / "bin"))
AUTOGRID_REPO = Path(os.environ.get("AUTOGRID_REPO", WORKSPACE / "autogrid"))
Q4_MODEL_DIR = Path(os.environ.get("SCION_Q4_MODEL_DIR", MODELS / "qwen38-flashnext-fp8"))
QWEN35_RELEASE_GGUF = os.environ.get("SCION_RELEASE_GGUF", str(WORKSPACE / "qwen35-release.gguf"))
