from __future__ import absolute_import
import os
import h5py
import cgp

D = os.path.dirname(os.path.abspath(__file__)) + '/'

geometry_file = D + 'data/geometry.h5'
results_file = D + 'data/geometry-results.h5'


def dataset_name(command, cell_order=None, label=None):
    name = command
    if cell_order is not None:
        name = '%s/cell-order-%d' % (name, cell_order)
    if label is not None:
        name = '%s/%d' %  (name, label)
    return name



f = h5py.File(results_file, 'w')

cgp_reader = cgp.GeometryReader(geometry_file, verbose=False)

for cell_order in [0,1,2,3]:
    f.create_dataset(dataset_name('max-label', cell_order),
                     data=cgp_reader.maxLabel(cell_order))
for cell_order in [0,1,2,3]:
    for label in range(1, cgp_reader.maxLabel(cell_order)+1):
        f.create_dataset(dataset_name('size', cell_order, label),
                         data=cgp_reader.size(cell_order, label))
for cell_order in [0,1,2]:
    for label in range(1, cgp_reader.maxLabel(cell_order)+1):
        f.create_dataset(dataset_name('bounds', cell_order, label),
                         data=cgp_reader.bounds(cell_order, label))
for cell_order in [1,2,3]:
    for label in range(1, cgp_reader.maxLabel(cell_order)+1):
        f.create_dataset(dataset_name('bounded-by', cell_order, label),
                         data=cgp_reader.boundedBy(cell_order, label))
for cell_order in [1,2,3]:
    for label in range(1, cgp_reader.maxLabel(cell_order)+1):
        f.create_dataset(dataset_name('adjacent', cell_order, label),
                         data=cgp_reader.adjacent(cell_order, label))

for label in range(1, cgp_reader.maxLabel(0)+1):
    f.create_dataset(dataset_name('zero-set', None, label),
                     data=cgp_reader.zeroSet(label))
for label in range(1, cgp_reader.maxLabel(1)+1):
    f.create_dataset(dataset_name('one-set', None, label),
                     data=cgp_reader.oneSet(label))
for label in range(1, cgp_reader.maxLabel(2)+1):
    f.create_dataset(dataset_name('two-set', None, label),
                     data=cgp_reader.twoSet(label))
for label in range(1, cgp_reader.maxLabel(3)+1):
    f.create_dataset(dataset_name('three-set', None, label),
                     data=cgp_reader.threeSet(label),
                     compression='lzf')

for cell_order in [1,2,3]:
    for label in range(1, cgp_reader.maxLabel(cell_order)+1):
        f.create_dataset(dataset_name('topological-point-set', cell_order, label),
                         data=cgp_reader.topologicalPointSet(cell_order, label),
                         compression='lzf')

f.close()


