"""
Leaf Disease Detection with ESP32 Serial Communication
"""

import cv2
import uuid
import os
import sys
import serial
import time
from tkinter import Tk, filedialog


# ================= SERIAL INITIALIZATION =================

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
time.sleep(2)  # Allow ESP32 reset


def serialprint(x):
    x = x + "\n"
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    print("Sent to ESP32:", x.strip())


# ================= IMAGE PROCESSOR CLASS =================

class ImageProcessor:

    def __init__(self, detector, output_dir="uploaded_datasets"):
        self.detector = detector
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_image(self, image_path, save_result=True):

        print("\n" + "="*60)
        print(f"Processing: {os.path.basename(image_path)}")
        print("="*60)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")

        boxes, scores, class_ids = self.detector.detect_objects(image)
        annotated_image, predicted_class = self.detector.draw_detections(image, is_image=True)

        if save_result:
            filename = f"{uuid.uuid4()}.jpg"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, annotated_image)

        if predicted_class:
            print("FINAL RESULT:", predicted_class)
        else:
            print("FINAL RESULT: HEALTHY")

        return predicted_class, annotated_image


def select_image_file():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    root.destroy()
    return file_path


# ================= MAIN =================

def main():

    print("\n" + "="*60)
    print("LEAF DISEASE DETECTION + ESP32")
    print("="*60)

    try:
        from detector import PlantDiseaseDetector
        detector = PlantDiseaseDetector()
    except Exception as e:
        print("Detector Error:", e)
        return

    processor = ImageProcessor(detector)

    image_path = select_image_file()

    if not image_path:
        print("No image selected")
        return

    predicted_class, annotated_img = processor.process_image(image_path)

    # ===== SERIAL SEND (Same logic as animal project) =====
    if predicted_class:
        serialprint("DISEASE:" + predicted_class)
    else:
        serialprint("HEALTHY")

    # Show image
    cv2.imshow("Leaf Disease Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
