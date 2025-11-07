import cv2
from ultralytics import YOLO
import time
import numpy as np

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # small model, downloads automatically

# Motor simulation functions
def stop(): print("[MOTORS] stop")
def forward(): print("[MOTORS] forward")
def left(): print("[MOTORS] left")
def right(): print("[MOTORS] right")

cap = cv2.VideoCapture(0)  # webcam
count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)[0]

        distance = 100  # default "no object"
        detected_objects = []

        for r in results.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            label = model.names[cls_id]
            detected_objects.append(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Approximate distance by bounding box width
            obj_width = x2 - x1
            distance = min(distance, 100 - obj_width)

        # Traffic light detection via color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red
        mask_red1 = cv2.inRange(hsv, (0,120,70), (10,255,255))
        mask_red2 = cv2.inRange(hsv, (170,120,70), (180,255,255))
        mask_red = mask_red1 + mask_red2
        if cv2.countNonZero(mask_red) > 50:
            cv2.putText(frame, "RED LIGHT", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        # Orange/Yellow
        mask_orange = cv2.inRange(hsv, (15,120,120), (35,255,255))
        if cv2.countNonZero(mask_orange) > 50:
            cv2.putText(frame, "ORANGE LIGHT", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255),2)
        # Green
        mask_green = cv2.inRange(hsv, (40,50,50), (90,255,255))
        if cv2.countNonZero(mask_green) > 50:
            cv2.putText(frame, "GREEN LIGHT", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        print(f"[SENSOR] Distance: {distance} cm | Objects: {detected_objects}")

        # Motor logic
        flag = 0
        if distance < 25:
            count += 1
            stop()
            time.sleep(0.5)
            if (count % 3 == 1) and (flag == 0):
                right()
                flag = 1
            else:
                left()
                flag = 0
            time.sleep(0.5)
            stop()
        else:
            forward()
            flag = 0

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting simulation...")
            break

except KeyboardInterrupt:
    print("Simulation interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and GUI closed")
