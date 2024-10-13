import open3d
import numpy as np

class PointcloudRegion():
    def __init__(self, pointcloud):
        self._pointcloud = pointcloud
        self._regions = []
        self._valid_indices: set[int] = set()
        self._updated = False

    def add_region(self, min_bound: np.ndarray, max_bound: np.ndarray) -> None:
        self._regions.append(open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
        self._updated = False

    def update_valid_points(self) -> None:
        self._valid_indices = set()
        for region in self._regions:
            self._valid_indices.update(region.get_point_indices_within_bounding_box(self._pointcloud.points))
        self._updated = True

    def contains(self, point: np.ndarray) -> bool:
        if len(self._regions) == 0:
            return True

        if isinstance(point.size, int) or len(point.size) == 1:
            point = point.reshape(1, 3)

        for region in self._regions:
            if len(region.get_point_indices_within_bounding_box(open3d.utility.Vector3dVector(point))):
                return True
        return False

    @property
    def pointcloud(self) -> np.ndarray:
        if not self._updated:
            self.update_valid_points()

        if len(self._valid_indices) == 0:
            return self._pointcloud
        else:
            valid_indices = list(self._valid_indices)
            ret_val = open3d.geometry.PointCloud()
            ret_val.points.extend(np.asarray(self._pointcloud.points)[valid_indices, :])
            ret_val.normals.extend(np.asarray(self._pointcloud.normals)[valid_indices, :])
            return ret_val