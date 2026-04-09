# Robotics Vision & Control Pipeline

This repository contains a complete, three-stage robotics vision pipeline bridging 2D pixel perception, 3D spatial geometry, and closed-loop visual servoing control.

## 🛠️ Environment Setup & Global Dependencies

It is recommended to run this project in a Python 3.8+ environment. 

Install all required dependencies for the entire pipeline using the following command:

`pip install numpy opencv-python opencv-contrib-python matplotlib ultralytics open3d`

*Note: You **must** install `opencv-contrib-python` (not just `opencv-python`) to access the `cv2.aruco` modules required for Section C3.*

---

## 🚀 Pipeline Modules & Execution Guide

### Section C1: 2D Perception & Image Moments
This module extracts an object's binary mask using YOLOv11x-seg. It computes the 2D center of mass using spatial image moments and calculates the object's principal orientation axis using translation-invariant second-order central moments.

* **Specific Dependencies:** `ultralytics`, `opencv-python`, `numpy`
* **How to Run:**
  `python 2d_perception.py` 
  *(Note: Replace `2d_perception.py` with the exact name of your C1 script)*
* **Expected Output:** An annotated 2D image displaying the bounding box, a red centroid dot, and a green vector indicating the principal axis.

### Section C2: 3D Grasp Estimation
This module deprojects 2D YOLO masks into a 3D point cloud using a 16-bit depth image and the Camera Intrinsic Matrix. It applies the RANSAC algorithm to isolate the flattest planar surface on the object and calculates the surface normal vector to determine the optimal robotic gripper approach angle.

* **Specific Dependencies:** `ultralytics`, `open3d`, `numpy`, `opencv-python`
* **How to Run:**
  `python grasp_depth.py`
* **Expected Output:** An interactive Open3D window displaying the segmented point cloud. The isolated grasp surface is highlighted in red, with a blue vector arrow indicating the perpendicular approach angle. You can use your mouse to rotate the 3D scene.

### Section C3: Image-Based Visual Servoing (IBVS)
This module simulates a closed-loop tracking system. It detects an off-center ArUco marker, calculates the pixel error from the screen's dead-center, and applies a proportional controller (Gain λ = 0.5) to simulate camera velocity via an Affine translation matrix.

* **Specific Dependencies:** `opencv-contrib-python`, `numpy`, `matplotlib`
* **How to Run:**
  This section requires a two-step execution.
  
  **Step 1:** Generate the synthetic ArUco tracking environment.
  `python generate_scene.py`
  *(This will generate `aruco_scene.png` in your directory).*

  **Step 2:** Run the visual servoing simulation loop.
  `python visual_servo.py`
* **Expected Output:** A live OpenCV window showing the camera panning to center the target, drawing a trajectory trail. Once the error drops below 5 pixels, a Matplotlib graph will appear demonstrating the critically damped, exponential decay of the tracking error.
