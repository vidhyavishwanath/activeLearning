"""
perception_server.py
--------------------
Runs on your Mac. Receives raw RGB frames from Pepper over TCP,
runs cv2 shape detection, and sends JSON results back.

Usage:
    python3 perception_server.py

Find your Mac's IP (what to put in MAC_IP on Pepper's side):
    ipconfig getifaddr en0
"""

import socket
import struct
import json
import cv2
import numpy as np

HOST = "0.0.0.0"  # listen on all interfaces
PORT = 9600

# ---------------------------------------------------------------------------
# Shape detection (cv2 version, runs on Mac)
# ---------------------------------------------------------------------------

COLOR_RANGES = {
    'red':    ([0,   120,  70], [10,  255, 255]),
    'blue':   ([100, 150,  50], [130, 255, 255]),
    'green':  ([40,   70,  50], [80,  255, 255]),
    'yellow': ([20,  100, 100], [35,  255, 255]),
    'orange': ([10,  150,  80], [20,  255, 255]),
    'purple': ([125,  40,  40], [160, 255, 255]),
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
        if perimeter == 0:
            return 'unknown'
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return 'circle' if circularity > 0.75 else 'unknown'


def detect_shapes(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    shapes = []
    h, w = frame_bgr.shape[:2]

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 400:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            label = get_shape_label(cnt)
            if label == 'unknown':
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            shapes.append({
                'label':   label,
                'color':   color_name,
                'cx':      cx,
                'cy':      cy,
                'cx_norm': cx / float(w),
                'cy_norm': cy / float(h),
                'left':    x,
                'right':   x + bw,
                'top':     y,
                'bottom':  y + bh,
                'width':   bw,
                'height':  bh,
            })

    return shapes


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

def recv_all(sock, n):
    """Receive exactly n bytes."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("Connection closed mid-receive")
        data += chunk
    return data


def handle_client(conn, addr):
    print("Connection from {}".format(addr))
    try:
        # Receive header: width (4 bytes) + height (4 bytes)
        header = recv_all(conn, 8)
        w, h = struct.unpack("!II", header)
        print("Frame size: {}x{}".format(w, h))

        # Receive raw RGB frame
        raw = recv_all(conn, w * h * 3)
        frame_rgb = np.array(bytearray(raw), dtype=np.uint8).reshape((h, w, 3))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Save for visual debugging
        cv2.imwrite("last_frame.jpg", frame_bgr)
        print("Saved last_frame.jpg")

        # Run detection
        shapes = detect_shapes(frame_bgr)
        print("Detected {} shapes: {}".format(len(shapes), [s['color'] + ' ' + s['label'] for s in shapes]))

        # Send back JSON
        payload = json.dumps(shapes).encode("utf-8")
        conn.sendall(struct.pack("!I", len(payload)))
        conn.sendall(payload)

    except Exception as e:
        print("Error: {}".format(e))
        # Send empty result on error
        payload = json.dumps([]).encode("utf-8")
        conn.sendall(struct.pack("!I", len(payload)))
        conn.sendall(payload)
    finally:
        conn.close()


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print("Perception server listening on port {}".format(PORT))
    print("Find your Mac IP with:  ipconfig getifaddr en0")
    print("Set MAC_IP in pepper_controller.py to that address\n")

    while True:
        conn, addr = server.accept()
        handle_client(conn, addr)


if __name__ == "__main__":
    main()
