import cv2
import numpy as np

# Tune these HSV ranges for your actual block colors
COLOR_RANGES = {
    'red':    ([0, 120, 70],   [10, 255, 255]),
    'blue':   ([100, 150, 50], [130, 255, 255]),
    'green':  ([40, 70, 50],   [80, 255, 255]),
    'yellow': ([20, 100, 100], [35, 255, 255]),
}

def get_shape_label(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    n = len(approx)

    if n == 3:
        return 'triangle'
    elif n == 4:
        return 'square'
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return 'circle' if circularity > 0.75 else 'unknown'

def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shapes = []

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            label = get_shape_label(cnt)
            if label == 'unknown':
                continue

            # Normalize coordinates to 0-1 for scale invariance
            h, w = frame.shape[:2]
            shapes.append({
                'label': label,
                'color': color_name,
                'cx': cx,
                'cy': cy,
                'cx_norm': cx / w,
                'cy_norm': cy / h,
            })

    return shapes


"""
FOR ROS2 reference

this is an example shapes array output:
[
    {
        'label': 'triangle',
        'color': 'red',
        'cx': 320,
        'cy': 215,
        'cx_norm': 0.5,
        'cy_norm': 0.448
    },
    {
        'label': 'square',
        'color': 'blue',
        'cx': 180,
        'cy': 310,
        'cx_norm': 0.281,
        'cy_norm': 0.646
    }
]

"""