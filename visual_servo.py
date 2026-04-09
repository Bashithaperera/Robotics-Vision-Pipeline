import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = "aruco_scene.png"
LAMBDA_GAIN = 0.5 
TARGET_U = 320 # Center X of a 640x480 image
TARGET_V = 240 # Center Y of a 640x480 image

# WHY LAMBDA = 0.5?
# In a proportional controller the gain (lambda) determines the step size.
# If lambda = 1.0, the system attempts to eliminate 100% of the error in a single frame which often leads to overshoot and infinite oscillation in discrete time digital systems.
# If lambda is too small (ex: 0.05), the convergence is extremely slow.
# A gain of lambda = 0.5 means the camera covers exactly 50% of the remaining distance to the target every frame. 
# This creates a mathematically smooth exponential decay (critically damped convergence) with absolutely zero overshoot.

# loading the base Aruco marker
base_world_img = cv2.imread(IMAGE_PATH)
if base_world_img is None:
    print("Error: Could not load aruco_scene.png")
    exit()

# State variables
camera_x, camera_y = 0.0, 0.0
trajectory_points = []
error_magnitudes = []
iteration = 0

print("Starting visual servoing simulation")

while True:
    iteration += 1
    
    # Camera motion simulation through pixel tranformation
    # If the camera moves right (+X) the world appears to move left (-X) in the image frame.
    translation_matrix = np.float32([[1, 0, -camera_x], [0, 1, -camera_y]])        # applying an Affine translation matrix to shift the base pixels opposite to camera position.
    frame = cv2.warpAffine(base_world_img, translation_matrix, (640, 480), borderValue=(50, 50, 50))
    
    # 2.Detecting the aruco makrer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #checking ArUco detection logic 
    try: 
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError: 
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, params)

    # if we lost the target stop the simulation
    if not corners:
        print("Target lost. Simulation aborted.")
        break

    # 3. Computing centroid and error
    c = corners[0][0]       #averaging the 4 corners of the marker to find its exact center
    u_centroid = int(np.mean(c[:, 0]))
    v_centroid = int(np.mean(c[:, 1]))
    
    #recording the current centroid for drawing the trajectory trail
    trajectory_points.append((u_centroid, v_centroid))

    #calculating image plane error ( e = Target - Centroid)
    e_u = TARGET_U - u_centroid
    e_v = TARGET_V - v_centroid
    
    # calculating Euclidean error 
    error_mag = np.linalg.norm([e_u, e_v])
    error_magnitudes.append(error_mag)

    # 4. Simulation termination condition
    if error_mag < 5.0:
        print(f"\nSUCCESS: Target Centered in {iteration} iterations!")
        print(f"Final Error Magnitude: {error_mag:.2f} pixels")
        break

    # 5. Proportionala controller
    # velocity = -lambda * e
    v_u = -LAMBDA_GAIN * e_u
    v_v = -LAMBDA_GAIN * e_v

    #applying velocity to camera position (camera motion opposite to the pixel motion)
    camera_x += v_u
    camera_y += v_v

    # 6. Visualization
    cv2.drawMarker(frame, (TARGET_U, TARGET_V), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)     # Draw crosshairs at the target center (320, 240)  
    cv2.circle(frame, (u_centroid, v_centroid), 5, (0, 255, 0), -1)     # Draw current centroid of the ArUco marker


    # drawing the the path the marker took across the screen (convergence trajectoory)
    for i in range(1, len(trajectory_points)):
        cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (255, 0, 0), 2)

    #displaying the simulation 
    cv2.putText(frame, f"Error Mag: {error_mag:.1f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Iter: {iteration}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Visual Servoing Simulation", frame)
    cv2.waitKey(500) 

cv2.destroyAllWindows()

# Error magnitude vs frame plot
print("Plotting error convergence curve...")
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_magnitudes) + 1), error_magnitudes, marker='o', linestyle='-', color='b')
plt.title("Visual Servoing: Error Magnitude vs Frame")
plt.xlabel("Frame (Iteration)")
plt.ylabel("Error Magnitude (pixels)")
plt.axhline(y=5, color='r', linestyle='--', label="Termination Threshold (5 px)")
plt.grid(True)
plt.legend()
plt.show()