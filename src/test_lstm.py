import pytest
from util import calcGain
from classes import Normalization
from util import getData


def test_calc():
    assert calcGain(-1, -2) == -1
    assert calcGain(-1, 1) == 2
    assert calcGain(1, 2) == -1
    assert calcGain(1, -1) == -2


def test_normalization():
    pass


def test_getData():
    data = getData('USDT-BTC', 100)
    print(data)
    assert len(data[0]["data"]) == 100
