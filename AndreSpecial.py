import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.stats import itemfreq

# Функция использует алгоритм кластеризации K-mean, чтобы найти доминирующий цвет
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

# realsense set up
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # realsense get depth and color frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        # cv transformation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 37)
        # detect
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=180, param2=40, minRadius=1, maxRadius=200)


        if not circles is None:
            circles = np.uint16(np.around(circles))
            max_r, max_i = 0, 0
            for i in range(len(circles[:, :, 2][0])):
                if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                    max_i = i
                    max_r = circles[:, :, 2][0][i]
            x, y, r = circles[:, :, :][0][max_i]
            if y > r and x > r:
                # depth for camera
                distance = depth_frame.get_distance(x, y)
                print(f"depth: {distance:.2f} m")


                # determinating dominant color & determinating zone
                square = frame[y-r:y+r, x-r:x+r]
                # OpenCV BGR, 0 - blue, 1 - green, 2 - red
                dominant_color = get_dominant_color(square, 2)
                # red
                if dominant_color[2] > 100:
                # if dominant_color[2] > 130:
                    print("31")

                # blue
                elif dominant_color[0] > 80:
                    zone_0 = square[square.shape[0]*3//8:square.shape[0] * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                    zone_0_color = get_dominant_color(zone_0, 1)
                    zone_1 = square[square.shape[0]*1//8:square.shape[0] * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                    zone_1_color = get_dominant_color(zone_1, 1)
                    zone_2 = square[square.shape[0]*3//8:square.shape[0] * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                    zone_2_color = get_dominant_color(zone_2, 1)
                    if zone_1_color[2] < 60:
                        if sum(zone_0_color) > sum(zone_2_color):
                            print("413")
                        else:
                            print("412")
                    else:
                        if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                            print("411")
                        elif sum(zone_0_color) > sum(zone_2_color):
                            print("415")
                        else:
                            print("414")
                # nothing detected
                else:
                        print("not possible")
                # detected circles
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('RealSense', frame)
        if cv2.waitKey(1) & 0xFF == 27:
                break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
