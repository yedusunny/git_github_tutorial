"""
Core Detection Module
Handles YOLO model initialization and basic detection operations
Can be run independently or imported by other modules
"""

import cv2
from yolov8 import YOLOv8
import sys
import os


class PlantDiseaseDetector:
    """
    Plant Disease Detection using YOLOv8
    """
    
    def __init__(self, model_path="model/best.onnx", confidence_threshold=0.2, iou_threshold=0.3):
        """
        Initialize the YOLO detector with model parameters
        
        Args:
            model_path (str): Path to the ONNX model file
            confidence_threshold (float): Minimum confidence for detection
            iou_threshold (float): IOU threshold for non-max suppression
        """
        self.model_path = model_path
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.detector = YOLOv8(model_path, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  - Confidence threshold: {confidence_threshold}")
        print(f"  - IOU threshold: {iou_threshold}")
    
    def detect_objects(self, frame):
        """
        Perform object detection on a single frame
        
        Args:
            frame: OpenCV image/frame (numpy array)
            
        Returns:
            boxes: Bounding boxes coordinates
            scores: Confidence scores
            class_ids: Detected class IDs
        """
        boxes, scores, class_ids = self.detector(frame)
        return boxes, scores, class_ids
    
    def draw_detections(self, frame, is_image=False):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: OpenCV image/frame
            is_image (bool): Flag to indicate if input is static image
            
        Returns:
            annotated_frame: Frame with drawn detections
            predicted_class: Name of the detected class
        """
        annotated_frame, predicted_class = self.detector.draw_detections(frame, is_image)
        return annotated_frame, predicted_class
    
    def get_detector(self):
        """
        Get the underlying detector instance
        
        Returns:
            YOLOv8 detector instance
        """
        return self.detector


def main():
    """
    Standalone execution - Test the detector with a sample image
    """
    print("\n" + "="*60)
    print("PLANT DISEASE DETECTOR - STANDALONE MODE")
    print("="*60 + "\n")
    
    # Initialize detector
    try:
        detector = PlantDiseaseDetector()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Test with sample image
    print("\nEnter image path to test (or press Enter to skip): ")
    image_path = input().strip()
    
    if image_path and os.path.exists(image_path):
        print(f"\n🔍 Testing detection on: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print("✗ Error: Could not read image")
            return
        
        # Detect
        boxes, scores, class_ids = detector.detect_objects(img)
        annotated_img, predicted_class = detector.draw_detections(img, is_image=True)
        
        # Display results
        print(f"✓ Detection complete")
        print(f"  - Found {len(boxes)} objects")
        print(f"  - Predicted class: {predicted_class or 'None'}")
        
        # Show image
        cv2.imshow("Detection Test", annotated_img)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("ℹ No test image provided. Detector initialized successfully.")
    
    print("\n✓ Detector module test complete")


if __name__ == "__main__":
    main()