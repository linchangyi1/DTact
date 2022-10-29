import numpy as np
import cv2
import yaml
import time


class Camera:
    def __init__(self, cfg):
        camera_setting = cfg['camera_setting']
        camera_channel = camera_setting['camera_channel']
        resolution_list = camera_setting['resolution_list']
        resolution_option = camera_setting['resolution_option']
        raw_img_width = resolution_list[resolution_option][0]
        raw_img_height = resolution_list[resolution_option][1]
        fps = camera_setting['fps']
        self.cap = cv2.VideoCapture(camera_channel)
        if self.cap.isOpened():
            print('------Camera is open--------')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, raw_img_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, raw_img_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        chessboard_images_path = cfg['camera_calibration']['chessboard_images_path']
        parameter_saving_file = chessboard_images_path + cfg['camera_calibration']['parameter_saving_file']
        camera_yaml = open(parameter_saving_file, 'r+', encoding='utf-8')
        camera_parameters = yaml.load(camera_yaml, Loader=yaml.FullLoader)
        self.mtx = np.array(camera_parameters['camera_matrix'])
        self.dist = np.array(camera_parameters['dist_coeff'])
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist,(raw_img_height, raw_img_width), 1,(raw_img_height, raw_img_width))

        sensor_setting = cfg['sensor_setting']
        self.crop_img_height = sensor_setting['crop_size'][0]
        self.crop_img_width = sensor_setting['crop_size'][1]
        self.surface_center_row = sensor_setting['surface_center'][0]
        self.surface_center_col = sensor_setting['surface_center'][1]
        self.width_begin = int(self.surface_center_col - self.crop_img_width/ 2)
        self.width_end = int(self.surface_center_col + self.crop_img_width / 2)
        self.height_begin = int(self.surface_center_row - self.crop_img_height / 2)
        self.height_end = int(self.surface_center_row + self.crop_img_height / 2)

    def get_raw_image(self):
        return self.cap.read()[1]

    def rectify_image(self, img):
        img_rectify = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        return img_rectify

    def crop_image(self, img):
        return img[self.height_begin:self.height_end, self.width_begin:self.width_end]

    def rectify_crop_image(self, img):
        img = self.crop_image(self.rectify_image(img))
        return img

    def get_rectify_image(self):
        img = self.rectify_image(self.get_raw_image())
        return img

    def get_rectify_crop_image(self):
        img = self.crop_image(self.get_rectify_image())
        return img

    def get_raw_avg_image(self):
        global img
        while True:
            img = self.cap.read()[1]
            cv2.imshow('img', img)
            crop_show = self.crop_image(img)
            cv2.imshow('crop_img', crop_show)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                cv2.destroyWindow('crop_img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.cap.read()[1]
            img_add += raw_image
            time.sleep(0.1)
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_rectify_avg_image(self):
        global img
        while True:
            img = self.get_rectify_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_image()
            img_add += raw_image
            time.sleep(0.1)
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_rectify_crop_avg_image(self):
        global img
        while True:
            img = self.get_rectify_crop_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_crop_image()
            img_add += raw_image
            time.sleep(0.1)
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def img_list_avg_rectify(self, img_list):
        img_1 = cv2.imread(img_list[0])
        img_add = np.zeros_like(img_1, float)
        for img_path in img_list:
            img = cv2.imread(img_path)
            img_add += img
        img_avg = img_add / len(img_list)
        img_avg = img_avg.astype(np.uint8)
        # cv2.imshow('ref_img_avg', ref_img_avg)
        # cv2.waitKey()
        ref_img_avg = self.rectify_image(img_avg)
        return ref_img_avg

