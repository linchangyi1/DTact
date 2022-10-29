import cv2
import numpy as np
import yaml
from camear import Camera
from _2_Sensor_Calibration import SensorCalibration

if __name__ == '__main__':
    f = open("_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor_calibration = SensorCalibration(cfg)
    ref = sensor_calibration.camera.get_rectify_avg_image()
    press_board_radius = 4.0 # CHANGE IT TO YOURS
    key = -1
    while True:
        img = sensor_calibration.camera.get_rectify_avg_image()
        cv2.imshow('image_circle', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        else:
            diff_raw = ref - img
            diff_mask = (diff_raw < 150).astype(np.uint8)
            diff = diff_raw * diff_mask
            center, detect_radius_p = sensor_calibration.circle_detection(diff)
            pixmm = press_board_radius/detect_radius_p
            print('center_row: ', center[1])
            print('center_column: ', center[0])
            print('pixmm: ', pixmm)
            cv2.destroyWindow('image_circle')
