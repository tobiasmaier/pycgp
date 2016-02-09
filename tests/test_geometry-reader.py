import os
import numpy as np
import h5py
import pytest
from numpy.testing import assert_equal

import pycgp


D = os.path.dirname(os.path.abspath(__file__)) + '/'

geometry_file = D + 'data/geometry.h5'
results_file = D + 'data/geometry-results.h5'


@pytest.fixture
def cgp_reader():
    return pycgp.GeometryReader(geometry_file, verbose=False)


@pytest.fixture
def data():
    results = h5py.File(results_file, 'r')

    def read_data(command, cell_order=None, label=None):
        group = command
        if cell_order is not None:
            group = '%s/cell-order-%d' % (group, cell_order)
        if label is not None:
            group = '%s/%d' % (group, label)
        return np.array(results[group])
    return read_data


def test_max_label(cgp_reader, data):
    for cell_order in [0, 1, 2, 3]:
        assert_equal(cgp_reader.max_label(cell_order),
                     data('max-label', cell_order))
    with pytest.raises(AssertionError):
        cgp_reader.max_label(-1)
    with pytest.raises(AssertionError):
        cgp_reader.max_label(4)


def test_size(cgp_reader, data):
    for cell_order in [0, 1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.size(cell_order, label),
                         data('size', cell_order, label))
    with pytest.raises(AssertionError):
        cgp_reader.size(-1, 1)
    with pytest.raises(AssertionError):
        cgp_reader.size(4, 1)
    with pytest.raises(AssertionError):
        cgp_reader.size(2, 0)
    with pytest.raises(AssertionError):
        cgp_reader.size(2, cgp_reader.max_label(2)+1)


def test_shape(cgp_reader, data):
    assert_equal(cgp_reader.shape(), [50, 50, 50])


def test_bounds(cgp_reader, data):
    for cell_order in [0, 1, 2]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.bounds(cell_order, label),
                         np.sort(data('bounds', cell_order, label)))
    with pytest.raises(AssertionError):
        cgp_reader.bounds(-1, 1)
    with pytest.raises(AssertionError):
        cgp_reader.bounds(3, 1)
    with pytest.raises(AssertionError):
        cgp_reader.bounds(2, 0)
    with pytest.raises(AssertionError):
        cgp_reader.bounds(2, cgp_reader.max_label(2)+1)


def test_bounded_by(cgp_reader, data):
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.bounded_by(cell_order, label),
                         data('bounded-by', cell_order, label))
    with pytest.raises(AssertionError):
        cgp_reader.bounded_by(0, 1)
    with pytest.raises(AssertionError):
        cgp_reader.bounded_by(4, 1)
    with pytest.raises(AssertionError):
        cgp_reader.bounded_by(2, 0)
    with pytest.raises(AssertionError):
        cgp_reader.bounded_by(2, cgp_reader.max_label(2)+1)


def test_adjacent(cgp_reader, data):
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.adjacent(cell_order, label),
                         data('adjacent', cell_order, label))
    with pytest.raises(AssertionError):
        cgp_reader.adjacent(0, 1)
    with pytest.raises(AssertionError):
        cgp_reader.adjacent(4, 1)
    with pytest.raises(AssertionError):
        cgp_reader.adjacent(2, 0)
    with pytest.raises(AssertionError):
        cgp_reader.adjacent(2, cgp_reader.max_label(2)+1)


def test_zero_set(cgp_reader, data):
    for label in range(1, cgp_reader.max_label(0)+1):
        assert_equal(cgp_reader.zero_set(label).points,
                     data('zero-set', None, label).reshape((1, 3)))
    with pytest.raises(AssertionError):
        cgp_reader.zero_set(0)
    with pytest.raises(AssertionError):
        cgp_reader.zero_set(cgp_reader.max_label(0)+1)


def test_one_set(cgp_reader, data):
    for label in range(1, cgp_reader.max_label(1)+1):
        assert_equal(cgp_reader.one_set(label).points,
                     data('one-set', None, label))
    with pytest.raises(AssertionError):
        cgp_reader.one_set(0)
    with pytest.raises(AssertionError):
        cgp_reader.one_set(cgp_reader.max_label(1)+1)


def test_two_set(cgp_reader, data):
    for label in range(1, cgp_reader.max_label(2)+1):
        assert_equal(cgp_reader.two_set(label).points,
                     data('two-set', None, label))
    with pytest.raises(AssertionError):
        cgp_reader.two_set(0)
    with pytest.raises(AssertionError):
        cgp_reader.two_set(cgp_reader.max_label(2)+1)


def test_three_set(cgp_reader, data):
    for label in range(1, cgp_reader.max_label(3)+1):
        assert_equal(cgp_reader.three_set(label).points,
                     data('three-set', None, label))
    with pytest.raises(AssertionError):
        cgp_reader.three_set(0)
    with pytest.raises(AssertionError):
        cgp_reader.three_set(cgp_reader.max_label(3)+1)


def test_topological_point_set(cgp_reader, data):
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(
                    cgp_reader.topological_point_set(cell_order, label).points,
                    data('topological-point-set', cell_order, label))
    with pytest.raises(AssertionError):
        cgp_reader.topological_point_set(0, 1)
    with pytest.raises(AssertionError):
        cgp_reader.topological_point_set(4, 1)
    with pytest.raises(AssertionError):
        cgp_reader.topological_point_set(2, 0)
    with pytest.raises(AssertionError):
        cgp_reader.topological_point_set(2, cgp_reader.max_label(2)+1)
