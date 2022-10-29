import cv2
import yaml
from sensor import Sensor
from visualizer import Visualizer


if __name__ == '__main__':
    f = open("_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor = Sensor(cfg)
    visualizer = Visualizer(sensor.points)

    while sensor.cap.isOpened():
        img = cv2.flip(sensor.get_rectify_crop_image(), 1)
        cv2.imshow('RawImage', img)
        height_map = sensor.raw_image_2_height_map(img)
        depth_map = sensor.height_map_2_depth_map(height_map)
        cv2.imshow('DepthMap', depth_map)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if not visualizer.vis.poll_events():
            break
        else:
            points, gradients = sensor.height_map_2_point_cloud_gradients(height_map)
            visualizer.update(points, gradients)
