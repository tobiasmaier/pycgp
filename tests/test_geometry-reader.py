import os
import numpy as np
import h5py
from numpy.testing import assert_equal

import pycgp

D = os.path.dirname(os.path.abspath(__file__)) + '/'

geometry_file = D + 'data/geometry.h5'
results_file = D + 'data/geometry-results.h5'

results = h5py.File(results_file, 'r')
cgp_reader = pycgp.GeometryReader(geometry_file, verbose=False)


def data(command, cell_order=None, label=None):
    group = command
    if cell_order is not None:
        group = '%s/cell-order-%d' % (group, cell_order)
    if label is not None:
        group = '%s/%d' % (group, label)
    return np.array(results[group])


def test_max_label():
    for cell_order in [0, 1, 2, 3]:
        assert_equal(cgp_reader.max_label(cell_order),
                     data('max-label', cell_order))


def test_size():
    for cell_order in [0, 1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.size(cell_order, label),
                         data('size', cell_order, label))


def test_bounds():
    for cell_order in [0, 1, 2]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.bounds(cell_order, label),
                         np.sort(data('bounds', cell_order, label)))


def test_bounded_by():
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.bounded_by(cell_order, label),
                         data('bounded-by', cell_order, label))


def test_adjacent():
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.adjacent(cell_order, label),
                         data('adjacent', cell_order, label))


def test_zero_set():
    for label in range(1, cgp_reader.max_label(0)+1):
        assert_equal(cgp_reader.zero_set(label).points,
                     data('zero-set', None, label).reshape((1, 3)))


def test_one_set():
    for label in range(1, cgp_reader.max_label(1)+1):
        assert_equal(cgp_reader.one_set(label).points,
                     data('one-set', None, label))


def test_two_set():
    for label in range(1, cgp_reader.max_label(2)+1):
        assert_equal(cgp_reader.two_set(label).points,
                     data('two-set', None, label))


def test_three_set():
    for label in range(1, cgp_reader.max_label(3)+1):
        assert_equal(cgp_reader.three_set(label).points,
                     data('three-set', None, label))


def test_topological_point_set():
    for cell_order in [1, 2, 3]:
        for label in range(1, cgp_reader.max_label(cell_order)+1):
            assert_equal(cgp_reader.topological_point_set(cell_order, label).points,
                         data('topological-point-set', cell_order, label))


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
