import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

# loading the yolo model
model = YOLO('yolov8s.pt')

# Mouse callback function to print RGB values on mouse move
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Seting up the named window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('veh2.mp4')

# Read class names from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
tracker = Tracker()
cy1 = 200  # the line variable the vehicle in the range of these lines will be counted
cy2 = 350
offset = 10
vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Defining the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 637))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 637))

    # Performing prediction using the YOLO model
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
            w = x2 - x1
            h = y2 - y1
            bbox_list.append([x1, y1, w, h])

    bbox_id = tracker.update(bbox_list)

    for bbox in bbox_id:
        x1, y1, w, h, id = bbox
        x2 = x1 + w
        y2 = y1 + h
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Checking for vehicles moving down
        if cy1 - offset < cy < cy1 + offset:
            if id not in vh_down:
                vh_down[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = time.time() - vh_down[id]
            if id not in counter:
                counter.append(id)
                distance = 20  # meters
                speed_ms = distance / elapsed_time
                speed_kh = speed_ms * 3.6
                cv2.putText(frame, f'{int(speed_kh)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Checking for vehicles moving up
        if cy2 - offset < cy < cy2 + offset:
            if id not in vh_up:
                vh_up[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time = time.time() - vh_up[id]
            if id not in counter1:
                counter1.append(id)
                distance1 = 20  # meters
                speed_ms1 = distance1 / elapsed1_time
                speed_kh1 = speed_ms1 * 3.6
                cv2.putText(frame, f'{int(speed_kh1)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Drawing lines and labels
    cv2.line(frame, (0, cy1), (frame.shape[1], cy1), (255, 255, 255), 2)
    cv2.putText(frame, 'L1', (5, cy1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (0, cy2), (frame.shape[1], cy2), (255, 255, 255), 2)
    cv2.putText(frame, 'L2', (5, cy2 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Displaying vehicle counts
    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, f'Going Down: {d}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Going Up: {u}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

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
