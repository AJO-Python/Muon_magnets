import pytest
from modules.dipole_grid import make_dipole_grid, check_run_name, set_angles


@pytest.fixture(scope="session")
def make_grid():
    grid = make_dipole_grid(config_file="test_dipole_array", testing=True)
    print(grid)
    return grid


def test_make_dipole_grid(make_grid):
    assert
