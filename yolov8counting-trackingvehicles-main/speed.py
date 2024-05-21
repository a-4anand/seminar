import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker  # Assuming tracker.py is in the same directory
import time

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Mouse callback function to print RGB values on mouse move
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up the named window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('veh2.mp4')

# Read class names from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
count = 0
tracker = Tracker()
cy1 = 322
cy2 = 368
offset = 6
vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

# Main loop to process each frame of the video
while True:    
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 500))

    # Perform prediction using the YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
             
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
            bbox_list.append([x1, y1, w, h])
    
    bbox_id = tracker.update(bbox_list)
    
    for bbox in bbox_id:
        x3, y3, w, h, id = bbox
        x4, y4 = x3 + w, y3 + h
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        
        # Check for vehicles moving down
        if cy1 - offset < cy < cy1 + offset:
            vh_down[id] = time.time()
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = time.time() - vh_down[id]
            if id not in counter:
                counter.append(id)
                distance = 10  # meters
                speed_ms = distance / elapsed_time
                speed_kh = speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(speed_kh)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        
        # Check for vehicles moving up
        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = time.time()
            if id not in counter1:
                counter1.append(id)
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time = time.time() - vh_up[id]
            if id not in counter1:
                counter1.append(id)      
                distance1 = 10  # meters
                speed_ms1 = distance1 / elapsed1_time
                speed_kh1 = speed_ms1 * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(speed_kh1)} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw lines and labels
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
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

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
