import numpy as np
from itertools import product

from .cgp_config import coordinate_type


class TopologicalPointSet():
    def __init__(self, cell_order, points):
        self.cell_order = cell_order
        self.points = np.abs(points)
        self.sign = 1 - 2 * np.any(np.asarray(points) < 0, axis=1).astype(np.int8)

    def adjacent_voxels(self):
        return np.array(map(self.point_to_voxel, self.points))

    def size(self):
        return len(self.points)

    def point_to_voxel(self, point):
        c = [None]*3
        for i in range(3):
            if point[i] % 2 == 0:
                c[i] = int(point[i]/2),
            else:
                c[i] = (int(point[i]/2), int((point[i]+1)/2))

        return list(product(c[0], c[1], c[2]))
