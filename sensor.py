import numpy as np
import cv2
from camear import Camera

class Sensor(Camera):
    def __init__(self, cfg):
        super().__init__(cfg)
        sensor_calibration = cfg['sensor_calibration']
        calibration_images_path = sensor_calibration['calibration_images_path']
        GRAY_Height_list_path = calibration_images_path + sensor_calibration['GRAY_Height_list']
        self.GRAY_Height_list = np.load(GRAY_Height_list_path)
        self.max_index = len(self.GRAY_Height_list) - 1

        # parameters for height_map
        # ref = self.get_rectify_crop_avg_image()
        ref = cv2.flip(self.get_rectify_crop_avg_image(), 1)
        self.ref_GRAY = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        sensor_reconstruction = cfg['sensor_reconstruction']
        self.lighting_threshold = sensor_reconstruction['lighting_threshold']
        self.kernel_list = sensor_reconstruction['kernel_list']
        self.contact_gray_base = sensor_reconstruction['contact_gray_base']
        self.depth_k = sensor_reconstruction['depth_k']

        # parameters for point_cloud
        self.points = np.zeros([self.crop_img_width * self.crop_img_height, 3])
        self.X, self.Y = np.meshgrid(np.arange(self.crop_img_width), np.arange(self.crop_img_height))
        Z = np.zeros_like(self.X)
        self.points[:, 0] = np.ndarray.flatten(self.X) * cfg['sensor_setting']['Pixmm']
        self.points[:, 1] = -np.ndarray.flatten(self.Y) * cfg['sensor_setting']['Pixmm']
        self.points[:, 2] = np.ndarray.flatten(Z)

    def raw_image_2_height_map(self, img):
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff_raw = self.ref_GRAY - img_GRAY - self.lighting_threshold
        diff_mask = (diff_raw < 100).astype(np.uint8)
        diff = diff_raw * diff_mask + self.lighting_threshold
        diff[diff>self.max_index] = self.max_index
        diff = cv2.GaussianBlur(diff.astype(np.float32), (7, 7), 0).astype(int) # this filter can decrease the lighting_threshold to 2
        height_map = self.GRAY_Height_list[diff] - self.GRAY_Height_list[self.lighting_threshold]
        for kernel in self.kernel_list:
            height_map = cv2.GaussianBlur(height_map.astype(np.float32), (kernel, kernel), 0)
        return height_map

    def get_height_map(self):
        # img = self.get_rectify_crop_image()
        img = cv2.flip(self.get_rectify_crop_image(), 1)
        height_map = self.raw_image_2_height_map(img)
        return height_map

    def height_map_2_depth_map(self, height_map):
        contact_show = np.zeros_like(height_map)
        contact_show[height_map > 0] = self.contact_gray_base
        depth_map = height_map * self.depth_k + contact_show
        depth_map = depth_map.astype(np.uint8)
        return depth_map

    def get_depth_map(self):
        height_map = self.get_height_map()
        depth_map = self.height_map_2_depth_map(height_map)
        return depth_map

    def height_map_2_point_cloud(self, height_map):
        self.points[:, 2] = np.ndarray.flatten(height_map)
        return self.points

    def height_map_2_point_cloud_gradients(self, height_map):
        height_gradients = np.gradient(height_map)
        points = self.height_map_2_point_cloud(height_map)
        return points, height_gradients

    def get_point_cloud(self):
        height_map = self.get_height_map()
        points = self.height_map_2_point_cloud(height_map)
        return points

    def get_point_cloud_gradients(self):
        height_map = self.get_height_map()
        points, height_gradients = self.height_map_2_point_cloud_gradients(height_map)
        return points, height_gradients







