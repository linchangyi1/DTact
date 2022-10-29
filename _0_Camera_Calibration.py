import numpy as np
import cv2
import glob
import time
import yaml
import os


class CameraCalibration:
    def __init__(self, cfg):
        camera_setting = cfg['camera_setting']
        self.camera_channel = camera_setting['camera_channel']
        self.resolution_list = camera_setting['resolution_list']
        self.resolution_option = camera_setting['resolution_option']
        self.img_width = self.resolution_list[self.resolution_option][0]
        self.img_height = self.resolution_list[self.resolution_option][1]

        camera_calibration = cfg['camera_calibration']
        if not os.path.exists('./images'):
            os.makedirs('./images')
        self.chessboard_images_path = camera_calibration['chessboard_images_path']
        if not os.path.exists(self.chessboard_images_path):
            os.makedirs(self.chessboard_images_path)
        self.cam_clb_img_number = camera_calibration['cam_clb_img_number']
        self.prefix = camera_calibration['prefix']
        self.image_format = camera_calibration['image_format']
        self.square_size = camera_calibration['square_size']
        self.corners_row = camera_calibration['corners_row']
        self.corners_column = camera_calibration['corners_column']
        self.parameter_saving_file = self.chessboard_images_path+camera_calibration['parameter_saving_file']
        self.optimization = camera_calibration['optimization']

    def collect_images(self):
        cap = cv2.VideoCapture(self.camera_channel)
        if cap.isOpened():
            print('------Camera is open--------')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        print('Ready to collect {} images for calibration !'.format(self.cam_clb_img_number))
        i = 1
        print('Press "s" to save or "q" to quit!')
        while True:
            ret, image = cap.read()
            # print(ret)
            cv2.imshow('image', image)
            key = cv2.waitKey(1)
            if key == ord('q') or i > self.cam_clb_img_number:
                break
            elif key == ord('s'):
                cv2.imwrite(self.chessboard_images_path+'/'+self.prefix + str(i) + '.' + self.image_format, image)
                print('Saving ' + self.chessboard_images_path+'/'+ self.prefix + str(i) + '.' + self.image_format)
                i += 1
        cap.release()
        cv2.destroyAllWindows()

    def calibration_process(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # set the chessboard plane to Z=0
        objp = np.zeros((self.corners_row*self.corners_column, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corners_column, 0:self.corners_row].T.reshape(-1, 2)
        # Create real world coordinate. Use your metric.
        objp = objp * self.square_size
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # Directory path correction. Remove the last character if it is '/'
        dirpath = self.chessboard_images_path
        if dirpath[-1:] == '/':
            dirpath = dirpath[:-1]
        # Get the images
        images = glob.glob(dirpath+'/' + self.prefix + '*.' + self.image_format)
        print('Successfully load the images to calibrate!')
        gray_shape = (0, 0)
        img_shape = (0, 0)
        # Iterate through the pairs and find chessboard corners. Add them to arrays
        # If openCV can't find the corners in an image, we discard the image.
        for fname in images:
            print(f'processing img:{fname}')
            start = time.time()
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]
            img_shape = img.shape[:2]
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.corners_column, self.corners_row), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                # detect subpixel corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # just for visualization
                img = cv2.drawChessboardCorners(img, (self.corners_column, self.corners_row), corners2, ret)
                cv2.namedWindow('img', 0)
                # cv2.resizeWindow('img', 500, 500)
                cv2.imshow('img', img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                end = time.time()
                print(f'time for processing of image{fname}: {end - start}')

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

        if not self.optimization:
            mtx_save = mtx.tolist()
            dist_save = dist.tolist()
            data = {'camera_matrix': mtx_save, 'dist_coeff': dist_save}
            with open(self.parameter_saving_file, 'w') as file:
                yaml.dump(data, file)
            print("Calibration is finished. RMS: ", ret)
            return [ret, mtx, dist, rvecs, tvecs]
        else:
            # optimization to get new and more accurate parameters
            img_h, img_w = img_shape
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
            newcameramtx_save = newcameramtx.tolist()
            dist_save = dist.tolist()
            data = {'camera_matrix': newcameramtx_save, 'dist_coeff': dist_save}
            with open(self.parameter_saving_file, 'w') as file:
                yaml.dump(data, file)
            print(roi)
            print("Calibration is finished. RMS: ", ret)
            return [ret, newcameramtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    f = open("_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cameracalibration = CameraCalibration(cfg)
    cameracalibration.collect_images()
    cameracalibration.calibration_process()
