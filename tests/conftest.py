import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def d():
    from cgm.sim import simulate
    return simulate()
