"""
REAL-TIME Leaf Disease Detection with ESP32 Serial Communication
Press 'q' to quit
"""

import cv2
import os
import sys
import serial
import time


# ================= SERIAL INITIALIZATION =================
arduino = serial.Serial(port='COM18', baudrate=9600, timeout=1)
time.sleep(2)  # Allow ESP32 reset


def serialprint(x):
    x = x + "\n"
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    print("Sent to ESP32:", x.strip())


# ================= IMAGE PROCESSOR CLASS =================

class ImageProcessor:

    def __init__(self, detector):
        self.detector = detector

    def process_frame(self, frame):

        boxes, scores, class_ids = self.detector.detect_objects(frame)
        annotated_image, predicted_class = self.detector.draw_detections(frame, is_image=True)

        if predicted_class:
            print("Detected:", predicted_class)

        return predicted_class, annotated_image


# ================= MAIN =================

def main():

    print("\n" + "="*60)
    print("REAL-TIME LEAF DISEASE DETECTION + ESP32")
    print("Press 'Q' to quit")
    print("="*60)

    try:
        from detector import PlantDiseaseDetector
        detector = PlantDiseaseDetector()
    except Exception as e:
        print("Detector Error:", e)
        return

    processor = ImageProcessor(detector)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    last_sent = ""
    last_send_time = 0
    cooldown = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_class, annotated_img = processor.process_frame(frame)

        current_time = time.time()

        # ===== SERIAL SEND CONTROL =====
        if predicted_class:
            if (predicted_class != last_sent) or (current_time - last_send_time > cooldown):
                serialprint("DISEASE:" + predicted_class)
                last_sent = predicted_class
                last_send_time = current_time
        else:
            # Optional: send healthy once
            pass

        cv2.imshow("Live Leaf Detection", annotated_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
    print("System Closed")


if __name__ == "__main__":
    main()
