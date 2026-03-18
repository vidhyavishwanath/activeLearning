import cv2
import numpy as np
import json
from shape_detection import detect_shapes  # your file

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found")
        break

    shapes = detect_shapes(frame)

    # Clear terminal each frame for readability
    print("\033[H\033[J", end="")

    if not shapes:
        print("No shapes detected")
    else:
        for s in shapes:
            print(f"{s['color']} {s['label']} at ({s['cx_norm']:.2f}, {s['cy_norm']:.2f})")

    # Optional: show the camera feed with annotations
    for s in shapes:
        cv2.circle(frame, (s['cx'], s['cy']), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{s['color']} {s['label']}", 
                    (s['cx'] - 40, s['cy'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Shape Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()