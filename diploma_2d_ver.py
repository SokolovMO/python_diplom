import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Define the minimum and maximum distance (in meters) for detecting traffic signs
MIN_DISTANCE = 0.5
MAX_DISTANCE = 5.0

while True:
    # Wait for a frame from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert the color frame to an OpenCV image
    color_image = np.asanyarray(color_frame.get_data())

    # Convert the depth frame to a depth map
    depth_image = np.asanyarray(depth_frame.get_data())

    # Apply a bilateral filter to the color image to reduce noise while preserving edges
    color_image = cv2.bilateralFilter(color_image, 9, 75, 75)

    # Apply a threshold to the depth map to create a binary mask for filtering out distant objects
    _, depth_mask = cv2.threshold(depth_image, MAX_DISTANCE, 255, cv2.THRESH_BINARY_INV)

    # Apply a median filter to the depth mask to remove small noise
    depth_mask = cv2.medianBlur(depth_mask, 5)

    # Apply the depth mask to the color image to filter out distant objects
    color_image = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

    # Convert the color image to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply a blur to the grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the grayscale image to create a binary image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and check if it meets the criteria for a traffic sign
    for contour in contours:
        # Get the area of the contour
        area = cv2.contourArea(contour)

        # Ignore contours that are too small or too large
        if area < 100 or area > 10000:
            continue

        # Get the distance to the contour from the RealSense camera
        x, y, w, h = cv2.boundingRect(contour)
        depth = depth_frame.get_distance(int(x + w/2), int(y + h/2))

        # Ignore contours that are too close or too far away
        if depth < MIN_DISTANCE or depth > MAX_DISTANCE:
            continue

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio of the bounding box
        aspect_ratio = w / h

        # Ignore contours that are not roughly square or rectangular
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            continue

        # Draw a rectangle around the contour on the color image
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	    # Draw a circle at the center of the contour on the depth map
    cv2.circle(depth_image, (int(x + w/2), int(y + h/2)), 5, (0, 0, 255), -1)

    # Print the distance to the contour on the color image
    cv2.putText(color_image, f"{depth:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the color and depth images
    cv2.imshow("Color", color_image)
    cv2.imshow("Depth", depth_image)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the RealSense camera and close all windows

pipeline.stop()
cv2.destroyAllWindows()
