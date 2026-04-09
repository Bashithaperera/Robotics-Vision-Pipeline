import cv2 
import numpy as np 
import open3d as o3d 
import os 
import math 
from ultralytics import YOLO 

#configuring absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RGB_PATH = os.path.join(SCRIPT_DIR, 'RGB1.png')     
DEPTH_PATH = os.path.join(SCRIPT_DIR, 'depth1.png') 

#camera inteinsic data (based on the dataset JSON)
# K = [[fx,  0, cx],
#      [ 0, fy, cy],
#      [ 0,  0,  1]]
FX = 1066.778  
FY = 1067.487  
CX = 312.9869  
CY = 241.3109  

# JSON depth scale is 0.1mm
DEPTH_SCALE = 10000.0  # converting to meters: raw value * 0.1 mm / 1000 mm = raw value / 10000.0

#YOLO SEGMENTATION
rgb_img = cv2.imread(RGB_PATH)
depth_img = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

if rgb_img is None or depth_img is None:
    print("Error: Could not load RGB or Depth image")
    exit()

#if the depth image has 3 channels slicing to keep only the first channel
if len(depth_img.shape) == 3:
    print("Warning: Depth image has 3 channels. Slicing to single channel")
    depth_img = depth_img[:, :, 0]

model = YOLO('yolo11x-seg.pt') # running YOLO to get the 2D object mask
results = model(rgb_img, conf=0.25)[0]

if results.masks is None:
    print("Error: No objects detected")
    exit()

#extracting the polygon contour 
contour = results.masks.xy[0].astype(np.int32)
binary_mask = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.uint8)  #blank binary mask matching the depth image dimensions
cv2.fillPoly(binary_mask, [contour], 1)

#CONVERTING 2D MASK TO 3D POINT CLOUD
print("Converting pixels to 3D space")

v_indices, u_indices = np.where(binary_mask == 1)   #finding all the pixel coordinates inside the object mask
raw_depths = depth_img[v_indices, u_indices]   # extracting raw depth values from the 16 bit depth map
z_values = raw_depths / DEPTH_SCALE #converting raw depth value into meters
valid = z_values > 0  #filtering out invalid depth values (<= 0)

if not np.any(valid):
    print("\nAll depth values inside the object mask are invalid (<= 0).")
    exit()

#applying the valid filter to our coordinate arrays
u = u_indices[valid]
v = v_indices[valid]
z = z_values[valid]

#applying the pinhole camera model inverse formula to generate physical X and Y in meters
x = (u - CX) * z / FX
y = (v - CY) * z / FY
points_3d = np.stack((x, y, z), axis=-1)  # stacking the X, Y, Z arrays into a single (N, 3) matrix

#creating an Open3D pointcloud object 
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# extracting matching RGB colors for the point cloud 
colors = rgb_img[v, u] 
colors = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
pcd.colors = o3d.utility.Vector3dVector(colors)

pcd = pcd.voxel_down_sample(voxel_size=0.005) # downsampling the cloud to smooth out sensor noise 


#RANSAC PLANE FITTING
print("Running RANSAC to find the top grasp surface")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000) # fitting RANSAC plane. distance_threshold set to 3cm to accommodate curved cylinder surfaces
A, B, C, D = plane_model             # extracting the normal vector [A, B, C] from the plane equation Ax + By + Cz + D = 0
normal_vector = np.array([A, B, C])

# to ensure the normal vector points towards the camera (negative Z in camera frame)
if C > 0:
    normal_vector = -normal_vector

# isolating the plane points (marking them in red))
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

outlier_cloud = pcd.select_by_index(inliers, invert=True)  # Isolating the rest of the object


# CENTROID & APPROACH METRICS
# Calculate the 3D center of mass of the object
centroid_3d = np.mean(np.asarray(pcd.points), axis=0)

# Calculate the approach angle relative to the camera's Z-axis using the dot product
approach_angle_rad = math.acos(abs(normal_vector[2]) / np.linalg.norm(normal_vector))
approach_angle_deg = math.degrees(approach_angle_rad)

#OPEN3D VISUALIZATION ---
print("Preparing 3D visualization.")

#centroid visualization
centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
centroid_sphere.translate(centroid_3d)
centroid_sphere.paint_uniform_color([0, 1, 0])

# Create a blue arrow to visually represent the surface normal (Approach Vector)
arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002, cone_radius=0.005, cylinder_height=0.05, cone_height=0.02)
arrow.paint_uniform_color([0, 0, 1])

# Calculate the rotation matrix to align the default vertical Open3D arrow to our normal vector
z_axis = np.array([0, 0, 1])
v_rot = np.cross(z_axis, normal_vector)
c_rot = np.dot(z_axis, normal_vector)
s_rot = np.linalg.norm(v_rot)

# Apply Rodrigues' rotation formula
if s_rot != 0:
    kmat = np.array([[0, -v_rot[2], v_rot[1]], 
                     [v_rot[2], 0, -v_rot[0]], 
                     [-v_rot[1], v_rot[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c_rot) / (s_rot ** 2))
    arrow.rotate(rotation_matrix, center=(0, 0, 0))

# Move the aligned arrow to sit at the centroid
arrow.translate(centroid_3d)

# MEtrics summary
print("\n=== GRASP PLANNING METRICS ===")
print(f"3D Centroid (X,Y,Z):  ({centroid_3d[0]:.4f}, {centroid_3d[1]:.4f}, {centroid_3d[2]:.4f}) meters")
print(f"Surface Normal Unit:  [{normal_vector[0]:.4f}, {normal_vector[1]:.4f}, {normal_vector[2]:.4f}]")
print(f"Grasp Approach Angle: {approach_angle_deg:.2f} degrees (relative to camera Z-axis)")
print("==============================\n")

# launching open3D
o3d.visualization.draw_geometries([outlier_cloud, inlier_cloud, centroid_sphere, arrow], 
                                  window_name="Grasp Region Estimation")