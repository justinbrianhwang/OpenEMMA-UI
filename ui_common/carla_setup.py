"""
CARLA 0.9.16 environment setup.

Auto-detects the CARLA installation within the project and adds
PythonAPI paths to sys.path. Must be imported before any carla imports.
"""

import os
import sys
import glob

# Project root (parent of ui_common/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CARLA may be installed in-repo or as a sibling of OpenEMMA-UI.
_CARLA_IN_REPO = os.path.join(PROJECT_ROOT, 'CARLA_0.9.16')
_CARLA_SIBLING = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'CARLA_0.9.16'))
CARLA_ROOT = next(
    (path for path in (_CARLA_IN_REPO, _CARLA_SIBLING) if os.path.isdir(path)),
    _CARLA_IN_REPO,
)
CARLA_PYTHONAPI = os.path.join(CARLA_ROOT, 'PythonAPI')
CARLA_AGENTS = os.path.join(CARLA_PYTHONAPI, 'carla')
CARLA_EXAMPLES = os.path.join(CARLA_PYTHONAPI, 'examples')


def setup_carla_paths():
    """Add CARLA PythonAPI paths to sys.path.

    Call this before importing carla or agents modules.
    Also attempts to install the carla wheel if not already available.
    """
    # Add PythonAPI paths
    for p in [CARLA_PYTHONAPI, CARLA_AGENTS, CARLA_EXAMPLES]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Check if carla module is importable
    try:
        import carla
        print(f'[CARLA] carla module loaded (version check: {CARLA_ROOT})')
    except ImportError:
        # Try installing from wheel
        dist_dir = os.path.join(CARLA_AGENTS, 'dist')
        wheels = glob.glob(os.path.join(dist_dir, 'carla-*.whl'))
        if wheels:
            wheel = wheels[0]
            print(f'[CARLA] Installing carla from wheel: {wheel}')
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel])
        else:
            raise ImportError(
                f'Cannot find carla module. '
                f'Expected wheel at {dist_dir} or carla package installed.'
            )

    return CARLA_ROOT
