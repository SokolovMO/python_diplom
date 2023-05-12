import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.stats import itemfreq

# K-mean clusterization to detect dominant colors
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]


# realsense set up
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# capture frames from realsense and detect depth and colors
def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    return depth_frame, color_frame

# processing frames & noise reduction
def process_frames(depth_frame, color_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=180, param2=40, minRadius=1, maxRadius=200)
    return circles, frame, depth_frame

# detect circles
def process_circles(circles, frame, depth_frame):
    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            process_detected_object(x, y, r, frame, depth_frame)

# output of the depth value and determination of the dominant color
def process_detected_object(x, y, r, frame, depth_frame):
    distance = depth_frame.get_distance(x, y)
    print(f"depth: {distance:.2f} m")
    square = frame[y-r:y+r, x-r:x+r]
    dominant_color = get_dominant_color(square, 2)
    process_dominant_color(dominant_color, square)

# checking the dominant color
def process_dominant_color(dominant_color, square):
    if dominant_color[2] > 100:
        print("31")
    elif dominant_color[0] > 50:
        process_blue_color(dominant_color, square)
    else:
        print("not possible")

# identification of arrows on blue signs
def process_blue_color(dominant_color, square):
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

# mapping circles to an image
def show_circles(circles, frame):
    if circles is not None:
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))  # преобразование в целые числа
            radius = int(i[2])  # преобразование радиуса в целое число
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), 3)
        cv2.imshow('RealSense', frame)

def main():
    pipeline = setup_realsense()
    try:
        while True:
            depth_frame, color_frame = get_frames(pipeline)
            circles, frame, depth_frame = process_frames(depth_frame, color_frame)
            process_circles(circles, frame, depth_frame)
            show_circles(circles, frame)
            cv2.imshow('RealSense', frame)  # отображение изображения вне зависимости от обнаружения кругов
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




