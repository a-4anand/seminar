import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

# Load YOLO model
model = YOLO('yolov8s.pt')

# Define mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Create named window and set mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video capture
cap = cv2.VideoCapture('veh2.mp4')

# Read class names from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n") 

# Initialize variables
tracker = Tracker()
cy1 = 300  # Increased distance between lines
cy2 = 400  # Increased distance between lines
offset = 6
vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame
    frame = cv2.resize(frame, (1020, 637))
    
    # Predict using YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Extract car bounding boxes
    car_bboxes = []
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            # Convert (x1, y1, x2, y2) to (x, y, w, h)
            w = x2 - x1
            h = y2 - y1
            car_bboxes.append([x1, y1, w, h])
    
    # Update tracker with car bounding boxes
    bbox_id = tracker.update(car_bboxes)
    
    # Draw tracked objects on frame
    for bbox in bbox_id:
        x1, y1, w, h, id = bbox
        x2 = x1 + w
        y2 = y1 + h
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Check for vehicles moving down
        if cy1 - offset < cy < cy1 + offset and id not in vh_down:
            vh_down[id] = time.time()
            print(f"Vehicle {id} started moving down at: {vh_down[id]}")  # Debug statement
            
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = time.time() - vh_down[id]
            if id not in counter:
                counter.append(id)
                distance = 20  # meters (Increased distance between lines)
                speed_ms = distance / elapsed_time
                speed_kh = speed_ms * 3.6
                print(f"Vehicle {id} moving down, elapsed time: {elapsed_time}, speed: {speed_kh} Km/h")  # Debug statement
                cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(speed_kh)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        
        # Check for vehicles moving up
        if cy2 - offset < cy < cy2 + offset and id not in vh_up:
            vh_up[id] = time.time()
            print(f"Vehicle {id} started moving up at: {vh_up[id]}")  # Debug statement
            
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time = time.time() - vh_up[id]
            if id not in counter1:
                counter1.append(id)      
                distance1 = 20  # meters (Increased distance between lines)
                speed_ms1 = distance1 / elapsed1_time
                speed_kh1 = speed_ms1 * 3.6
                print(f"Vehicle {id} moving up, elapsed time: {elapsed1_time}, speed: {speed_kh1} Km/h")  # Debug statement
                cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(speed_kh1)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw lines on frame
    cv2.line(frame, (0, cy1), (frame.shape[1], cy1), (255, 255, 255), 2)
    cv2.putText(frame, 'L1', (5, cy1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (0, cy2), (frame.shape[1], cy2), (255, 255, 255), 2)
    cv2.putText(frame, 'L2', (5, cy2 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    # Display vehicle counts
    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, f'going down: {d}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'going up: {u}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
