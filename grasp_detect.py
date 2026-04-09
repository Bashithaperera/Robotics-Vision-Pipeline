import cv2 
import numpy as np 
import os 
import math 
from ultralytics import YOLO 

#setting up the abbsolute path of the directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(SCRIPT_DIR, 'images') 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output') 

#checking if output dir exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#pre-trained YOLO model
model = YOLO('yolo11x-seg.pt') 

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpeg', '.jpg'))] #fetching all jpeg image files
image_files.sort()

summary_data = [] #empty list to generate summary

print(f"Processing {len(image_files)} images from: {INPUT_DIR}\n")

for img_name in image_files:
    
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    
    results = model(img, conf=0.25)[0] #confidence threshold at 25%
    annotated_img = results.plot()
    
    if results.masks is not None:
        
        for i in range(len(results.masks)):    #iterating through every detected object in the current image 
            
            box = results.boxes[i]  #bounding box data 
            label = model.names[int(box.cls[0])] #class label 
            conf = float(box.conf[0]) #obtaining the confidence score
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() #getting the bounding box coordinates
            bbox_area = (x2 - x1) * (y2 - y1) #calculating bounding box area
            contour = results.masks.xy[i].astype(np.int32) #getting the pixel coordinates of the segmentation mask contour 
            
            M = cv2.moments(contour)         #caalculating the spatial and central moments of the mask 
            if M["m00"] != 0:       # calculating the centroid (u, v) using the first order moments
                cx = int(M["m10"] / M["m00"])       #m00: area, m10: sum of x-coordinates of pixels, m01: sum of y-coordinates of pixels
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            if (M["mu20"] - M["mu02"]) != 0:                # calculating the principal axis angle using second-order central moments 
                angle_rad = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])      #m20: variance along x-axis, m02: variance along y-axis, m11: covariance of x and y 
            else:
                angle_rad = 0
            angle_deg = math.degrees(angle_rad) #rad to deg

            cv2.circle(annotated_img, (cx, cy), 5, (0, 0, 255), -1) #marking the centroid of the detection using a red circle

            #caalculating the endpoint for the principal axis arrow 
            line_len = 400
            end_x = int(cx + line_len * math.cos(angle_rad))
            end_y = int(cy + line_len * math.sin(angle_rad))            
            cv2.arrowedLine(annotated_img, (cx, cy), (end_x, end_y), (0, 255, 0), 5) #green arrow to represent the principal axis direction

            #storing the data as a formatted string for the final summary table
            row = f"{img_name:<12} | {label:<12} | {conf:<5.2f} | ({cx:^4}, {cy:^4}) | {angle_deg:<10.2f} | {bbox_area:.1f}"
            summary_data.append(row)

    # Save the final annotated image to the output folder
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), annotated_img)


#printing summary table on the terminal
print("\n" + "="*85)
print(f"{'Image':<12} | {'Label':<12} | {'Conf':<5} | {'Centroid(u,v)':<12} | {'Angle(deg)':<10} | {'BBox Area'}")
print("-" * 85)
for row in summary_data:
    print(row)
print("="*85)
print(f"\nSuccess: Annotated images saved to {OUTPUT_DIR}")