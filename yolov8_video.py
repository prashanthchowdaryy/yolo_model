import cv2
from ultralytics import YOLO

# ----------------------------
# Load YOLOv8 model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Input video path (FIXED)
# ----------------------------
video_path = r"C:\Users\Prashanth\Desktop\Naresh_it\AI\DeepLearning\CNN\YOLO\anatgiri_trip.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(" Error: Could not open video.")
    exit()

# ----------------------------
# Video writer (FIXED)
# ----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

print(" Processing video... Press 'q' to quit")

# ----------------------------
# Frame loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Video finished.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Video Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(" Output saved as output.mp4")