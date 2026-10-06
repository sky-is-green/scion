import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scion_paths  # noqa: E402

# The W3 probe drives the AUTOGRID fork's DeltaLoss ranker, so the fork has to
# be importable for its tests.  Override with AUTOGRID_REPO; the DeltaLoss tests
# skip (rather than fail) when the checkout is not there.
_AUTOGRID = scion_paths.AUTOGRID_REPO
if (_AUTOGRID / "autogrid_ext").is_dir():
    sys.path.insert(0, str(_AUTOGRID))
else:  # pragma: no cover - environment dependent
    _AUTOGRID = None
