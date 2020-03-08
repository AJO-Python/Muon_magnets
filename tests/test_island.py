import pytest
import numpy as np
from modules.island import Island


@pytest.fixture
def test_island():
    parameters = {
        "orientation": 0,
        "coord": [0, 0],
        "strength": 1,
        "size": (2, 2),
        "location": [0, 0]
    }
    island = Island(**parameters)
    return island


def test_init(test_island):
    assert isinstance(test_island, object)
    assert test_island.size == (2, 2)
    assert test_island.location == [0, 0]
    actual_init = [[-1, 1], [1, 1], [-1, -1], [1, -1]]
    for calc, acc in zip(test_island.corners, actual_init):
        assert set(calc) == set(acc)


def test_set_corners(test_island):
    # Test simple rotation
    test_island.orientation_r = np.pi / 2
    test_island.set_corners()
    actual_SR = list([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    corners = [[*c] for c in test_island.corners]
    for calc, acc in zip(corners, actual_SR):
        assert calc == pytest.approx(acc)


def test_rotate(test_island):
    # Test rotation about non-origin point
    test_island.location = [1, 1]
    test_island.orientation_r = np.pi / 2
    test_point = [1, 2]
    assert list(test_island.rotate(test_point)) == [0, 1]
