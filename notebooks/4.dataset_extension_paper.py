"""Compatibility entry point for paper diversity visualization.

The implementation lives in ``scripts/batch_diversity_visualization.py``.
Running this notebook-side script keeps the old filename convenient while
defaulting to ``dataset/train`` as the input root.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_diversity_visualization import main  # noqa: E402


def _inject_notebook_defaults(argv: list[str]) -> list[str]:
    """Let ``python notebooks/4.dataset_extension_paper.py`` work directly."""
    if '--input-root' not in argv:
        argv = [*argv, '--input-root', str(PROJECT_ROOT / 'dataset' / 'train')]
    return argv


if __name__ == '__main__':
    sys.argv = _inject_notebook_defaults(sys.argv)
    raise SystemExit(main())
