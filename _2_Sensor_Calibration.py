# collect images and calibrate the sensor
import numpy as np
import cv2
import glob
import yaml
import os
from camear import Camera


class SensorCalibration:
	def __init__(self, cfg):
		self.camera = Camera(cfg)
		self.Pixmm = cfg['sensor_setting']['Pixmm']  # .10577 #4.76/100 #0.0806 * 1.5 mm/pixel # 0.03298 = 3.40 / 103.0776
		sensor_calibration = cfg['sensor_calibration']
		if not os.path.exists('./images'):
			os.makedirs('./images')
		calibration_images_path = sensor_calibration['calibration_images_path']
		if not os.path.exists(calibration_images_path):
			os.makedirs(calibration_images_path)
		self.ref_images_path = calibration_images_path + sensor_calibration['ref_directory']
		self.sample_images_path = calibration_images_path + sensor_calibration['sample_directory']
		if not os.path.exists(self.ref_images_path):
			os.makedirs(self.ref_images_path)
		if not os.path.exists(self.sample_images_path):
			os.makedirs(self.sample_images_path)
		self.ref_number = sensor_calibration['ref_number']
		self.sample_number = sensor_calibration['sample_number']
		self.BallRad = sensor_calibration['BallRad']  # 10.00/2 #mm
		self.circle_detection_gray = sensor_calibration['circle_detect_gray']
		self.show_circle_detection = sensor_calibration['show_circle_detection']
		self.GRAY_Height_list_path = calibration_images_path + sensor_calibration['GRAY_Height_list']

	def collect_images(self):
		if self.ref_number:
			print('Reference image number: {}'.format(self.ref_number))
			for i in range(self.ref_number):
				ref = self.camera.get_raw_avg_image()
				cv2.imwrite(self.ref_images_path + '/ref_' + str(i + 1) + '.png', ref)
				print('saving ' + self.ref_images_path + '/ref_' + str(i + 1) + '.png')
		if self.sample_number:
			print('Sample image number: {}'.format(self.sample_number))
			for j in range(self.sample_number):
				sample = self.camera.get_raw_avg_image()
				cv2.imwrite(self.sample_images_path + '/sample_' + str(j + 1) + '.png', sample)
				print('saving ' + self.sample_images_path + '/sample_' + str(j + 1) + '.png')

	def circle_detection(self, diff):
		diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
		contact_mask = (diff_gray > self.circle_detection_gray).astype(np.uint8)
		contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		sorted_areas = np.sort(areas)
		if len(sorted_areas):
			cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
			(x, y), radius = cv2.minEnclosingCircle(cnt)
			center = (int(x), int(y))
			radius = int(radius)
			if self.show_circle_detection:
				key = -1
				print('If the detected circle is suitable, press the key "q" to continue!')
				while key != ord('q'):
					center = (int(x), int(y))
					radius = int(radius)
					circle_show = cv2.circle(np.array(diff), center, radius, (0, 255, 0), 1)
					circle_show[int(y), int(x)] = [255, 255, 255]
					cv2.imshow('contact', circle_show.astype(np.uint8))
					key = cv2.waitKey(0)
					if key == ord('w'):
						y -= 1
					elif key == ord('s'):
						y += 1
					elif key == ord('a'):
						x -= 1
					elif key == ord('d'):
						x += 1
					elif key == ord('m'):
						radius += 1
					elif key == ord('n'):
						radius -= 1
				cv2.destroyWindow('contact')
			return center, radius
		else:
			return (0, 0), 0

	def mapping_data_collection(self, img, ref, gray_list=None, depth_list=None):
		diff_raw = ref - img
		diff_mask = (diff_raw < 150).astype(np.uint8)
		diff = diff_raw * diff_mask
		cv2.imshow('ref', ref)
		cv2.imshow('img', img)
		cv2.imshow('diff', diff)
		center, detect_radius_p = self.circle_detection(diff)
		if detect_radius_p:
			x = np.linspace(0, diff.shape[0] - 1, diff.shape[0])  # [0, 479]
			y = np.linspace(0, diff.shape[1] - 1, diff.shape[1])  # [0, 639]
			xv, yv = np.meshgrid(y, x)
			xv = xv - center[0]
			yv = yv - center[1]
			rv = np.sqrt(xv ** 2 + yv ** 2)
			mask = (rv < detect_radius_p)
			temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.Pixmm ** 2
			height_map = (np.sqrt(self.BallRad ** 2 - temp) * mask - np.sqrt(self.BallRad ** 2 - (detect_radius_p * self.Pixmm) ** 2)) * mask
			height_map[np.isnan(height_map)] = 0
			diff_gray = (diff[::, ::, 0] + diff[::, ::, 1] + diff[::, ::, 2]) / 3
			diff_gray = self.camera.crop_image(diff_gray)
			height_map = self.camera.crop_image(height_map)
			count = 0
			for i in range(height_map.shape[0]):
				for j in range(height_map.shape[1]):
					if height_map[i, j] > 0:
						gray_list.append(diff_gray[i, j])
						depth_list.append(height_map[i, j])
						count += 1
			print('Sample points number: {}'.format(count))
			return gray_list, depth_list

	def get_GRAY_Height_list(self, gray_list, depth_list):
		GRAY_scope = int(gray_list.max())
		GRAY_Height_list = np.zeros(GRAY_scope + 1)
		for gray_number in range(GRAY_scope + 1):
			gray_height_sum = depth_list[gray_list == gray_number].sum()
			gray_height_num = (gray_list == gray_number).sum()
			if gray_height_num:
				GRAY_Height_list[gray_number] = gray_height_sum / gray_height_num
		for gray_number in range(GRAY_scope + 1):
			if GRAY_Height_list[gray_number] == 0:
				if not gray_number:
					min_index = gray_number - 1
					max_index = gray_number + 1
					for i in range(GRAY_scope - gray_number):
						if GRAY_Height_list[gray_number + 1 + i] != 0:
							max_index = gray_number + 1 + i
							break
					GRAY_Height_list[gray_number] = (GRAY_Height_list[max_index] - GRAY_Height_list[min_index]) / (max_index - min_index)
		return GRAY_Height_list

	def calibrate_process(self):
		print('Begin to calibrate')
		ref_img_list = glob.glob(f"{self.ref_images_path}/ref*.png")
		ref = self.camera.img_list_avg_rectify(ref_img_list)

		sample_list = sorted(glob.glob(f"{self.sample_images_path}/sample*.png"), key=os.path.getctime)
		global gray_list, depth_list
		gray_list = list()
		depth_list = list()

		for sample_path in sample_list:
			sample = cv2.imread(sample_path)
			sample = self.camera.rectify_image(sample)
			gray_list, depth_list = self.mapping_data_collection(sample, ref, gray_list=gray_list, depth_list=depth_list)
		gray_list = np.array(gray_list)
		depth_list = np.array(depth_list)
		GRAY_H_table = self.get_GRAY_Height_list(gray_list, depth_list)
		np.save(self.GRAY_Height_list_path, GRAY_H_table)


if __name__ == '__main__':
	f = open("_config.yaml", 'r+', encoding='utf-8')
	cfg = yaml.load(f, Loader=yaml.FullLoader)
	sensor_calibration = SensorCalibration(cfg)
	sensor_calibration.collect_images()
	sensor_calibration.calibrate_process()
