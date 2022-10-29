import numpy as np
import open3d
from open3d import *

class Visualizer:
    def __init__(self, points):
        self.init_visualizer(points)

    def init_visualizer(self, points):
        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='DTact',width=700,height=700)
        self.vis.add_geometry(self.pcd)
        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-10)
        # print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.9)
        self.ctr.rotate(0, -400)  # mouse drag in x-axis, y-axis
        self.vis.update_renderer()

    def update(self, points, gradients):
        dx, dy = gradients
        dx, dy = dx * (-10), dy * 10
        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([points.shape[0], 3])
        for _ in range(3):
            colors[:, _] = np_colors
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)
        try:
            self.vis.update_geometry()
        except:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

