"""Path defaults for the dense harness. Every default is environment-overridable.

SCION_WORKSPACE  workspace root (default: the parent directory of this repo)
SCION_MODELS     model directory (default: $SCION_WORKSPACE/models)
SCION_CLEF_MODEL clef-flash-ternary directory (default: $SCION_MODELS/clef-flash-ternary)
SCION_HIVEBENCH  sibling hivebench checkout (default: $SCION_WORKSPACE/hivebench)
LLAMA_BIN        llama.cpp build bin directory (default: ~/llama.cpp/build/bin)
GGUF_PY          llama.cpp gguf-py directory (default: ~/llama.cpp/gguf-py)
SCION_BRIDGE     clef bridge executable (default: $SCION_HIVEBENCH/tools/clef-bridge/clef_embed)
SCION_STORAGE    scratch/storage directory (default: $SCION_WORKSPACE/storage)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("SCION_WORKSPACE", REPO.parent))
MODELS = Path(os.environ.get("SCION_MODELS", WORKSPACE / "models"))
CLEF_MODEL = Path(os.environ.get("SCION_CLEF_MODEL", MODELS / "clef-flash-ternary"))
HIVEBENCH = Path(os.environ.get("SCION_HIVEBENCH", WORKSPACE / "hivebench"))
HIVE = HIVEBENCH / "experiments" / "cascade"
LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", Path.home() / "llama.cpp" / "build" / "bin"))
GGUF_PY = Path(os.environ.get("GGUF_PY", Path.home() / "llama.cpp" / "gguf-py"))
BRIDGE = os.environ.get("SCION_BRIDGE", str(HIVEBENCH / "tools" / "clef-bridge" / "clef_embed"))
STORAGE = Path(os.environ.get("SCION_STORAGE", WORKSPACE / "storage"))
