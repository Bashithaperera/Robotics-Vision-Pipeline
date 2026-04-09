import cv2
import numpy as np
import os

# --- PATH FIX ---
# Get the absolute path of the folder this script is in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "aruco_scene.png")

# Create a blank white 640x480 canvas
img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Draw a grid background to make camera motion visible
for i in range(0, 640, 40): cv2.line(img, (i, 0), (i, 480), (220, 220, 220), 1)
for i in range(0, 480, 40): cv2.line(img, (0, i), (640, i), (220, 220, 220), 1)

# Generate a standard 4x4 ArUco marker
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 80)
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.drawMarker(aruco_dict, 0, 80)

# Place the marker off-center (bottom right quadrant)
x_offset, y_offset = 500, 350
img[y_offset:y_offset+80, x_offset:x_offset+80] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

# Save using the absolute path
cv2.imwrite(OUTPUT_PATH, img)
print(f"Success: Image saved exactly to -> {OUTPUT_PATH}")