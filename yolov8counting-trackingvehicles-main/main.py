import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

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
cy1 = 500
cy2 = 550
offset = 6

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame
    frame = cv2.resize(frame, (1020, 500))
    
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
            car_bboxes.append([x1, y1, x2, y2])
    
    # Update tracker with car bounding boxes
    bbox_id = tracker.update(car_bboxes)
    
    # Draw tracked objects on frame
    for bbox in bbox_id:
        x1, y1, x2, y2, id = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        
    # Draw lines on frame
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow("RGB", frame)
    
    # Check for key press event
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
