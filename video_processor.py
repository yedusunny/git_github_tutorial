"""
Video Processing Module
Handles detection on video files
Can be run independently or imported by other modules
"""

import cv2
import uuid
import os
import sys
from tkinter import Tk, filedialog


class VideoProcessor:
    """
    Process video files for plant disease detection
    """
    
    def __init__(self, detector, output_dir="uploaded_datasets"):
        """
        Initialize video processor with detector instance
        
        Args:
            detector: PlantDiseaseDetector instance
            output_dir (str): Directory to save results
        """
        self.detector = detector
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}")
    
    def process_video(self, video_path, save_snapshots=True, display_live=True):
        """
        Process video file for disease detection
        
        Args:
            video_path (str): Path to video file
            save_snapshots (bool): Save frames when disease detected
            display_live (bool): Show live detection window
            
        Returns:
            snapshots: List of saved snapshot paths
            detections: List of detected classes throughout video
        """
        print(f"\n{'='*60}")
        print(f"🎥 Processing video: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        
        print(f"\n📊 VIDEO PROPERTIES:")
        print(f"   Resolution: {width}x{height} pixels")
        print(f"   Total frames: {total_frames}")
        print(f"   Frame rate: {fps} FPS")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   File size: {file_size:.2f} MB")
        
        # Setup display window if needed
        if display_live:
            window_name = "Video Detection - Plant Disease"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Auto-resize window based on video dimensions
            screen_width = 1920
            screen_height = 1080
            max_width = int(screen_width * 0.7)
            max_height = int(screen_height * 0.7)
            
            width_scale = max_width / width
            height_scale = max_height / height
            scale = min(width_scale, height_scale, 1.0)
            
            window_width = int(width * scale)
            window_height = int(height * scale)
            
            cv2.resizeWindow(window_name, window_width, window_height)
            print(f"   Display window: {window_width}x{window_height} pixels")
            if scale < 1.0:
                print(f"   Scaled to: {scale*100:.1f}% of original size")
        
        snapshots = []
        detections = []
        detection_count = 0
        frame_count = 0
        last_detection = None
        
        print(f"\n{'='*60}")
        print("▶️  PROCESSING VIDEO...")
        print(f"{'='*60}")
        print("⌨️  Controls: 'q' to quit | 's' to save snapshot | ESC to exit")
        print(f"{'='*60}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection
            boxes, scores, class_ids = self.detector.detect_objects(frame)
            annotated_frame, predicted_class = self.detector.draw_detections(frame, is_image=False)
            
            # Track detections
            if predicted_class:
                detection_count += 1
                detections.append({
                    'frame': frame_count,
                    'class': predicted_class,
                    'time': frame_count / fps
                })
                
                # Log new detections (not repeats)
                if predicted_class != last_detection:
                    timestamp = frame_count / fps
                    print(f"🔍 Frame {frame_count:6d} ({timestamp:7.2f}s): {predicted_class}")
                    last_detection = predicted_class
            else:
                last_detection = None
            
            # Display live feed
            if display_live:
                # Create info overlay
                overlay_height = 80
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # Add frame info
                progress = (frame_count / total_frames) * 100
                info_text1 = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
                info_text2 = f"Time: {frame_count/fps:.2f}s / {duration:.2f}s"
                info_text3 = f"Detection: {predicted_class or 'None'} | Found: {detection_count}"
                
                # Adaptive font size based on frame size
                if min(height, width) < 640:
                    font_scale = 0.4
                    thickness = 1
                else:
                    font_scale = 0.6
                    thickness = 2
                
                cv2.putText(annotated_frame, info_text1, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.putText(annotated_frame, info_text2, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.putText(annotated_frame, info_text3, (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q' or ESC
            if key == ord('q') or key == 27:
                print(f"\n⏸️  Processing stopped by user at frame {frame_count}")
                break
            
            # Save snapshot on 's'
            elif key == ord('s'):
                snapshot_path = self._save_snapshot(annotated_frame, frame_count, predicted_class)
                snapshots.append(snapshot_path)
                print(f"📸 Snapshot saved: {os.path.basename(snapshot_path)}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print(f"\n{'='*60}")
        print("✅ VIDEO PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 STATISTICS:")
        print(f"   Processed frames: {frame_count}/{total_frames}")
        print(f"   Frames with detection: {detection_count} ({detection_count/frame_count*100:.1f}%)")
        print(f"   Unique detections: {len(set(d['class'] for d in detections))}")
        print(f"   Snapshots saved: {len(snapshots)}")
        
        # Show detection breakdown
        if detections:
            class_counts = {}
            for det in detections:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print(f"\n📋 DETECTION BREAKDOWN:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / detection_count) * 100
                print(f"   • {cls}: {count} frames ({percentage:.1f}%)")
        
        print(f"{'='*60}\n")
        
        return snapshots, detections
    
    def _save_snapshot(self, frame, frame_number, detection_class):
        """
        Save a frame snapshot with descriptive filename
        
        Args:
            frame: OpenCV frame to save
            frame_number: Frame number in video
            detection_class: Detected class name
            
        Returns:
            snapshot_path: Path to saved snapshot
        """
        # Create descriptive filename
        class_name = detection_class.replace(' ', '_') if detection_class else 'no_detection'
        snapshot_filename = f"snapshot_frame{frame_number}_{class_name}_{uuid.uuid4().hex[:8]}.jpg"
        snapshot_path = os.path.join(self.output_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
        return snapshot_path
    
    def save_detection_report(self, detections, output_file="detection_report.txt"):
        """
        Save detection results to a text report
        
        Args:
            detections (list): List of detection dictionaries
            output_file (str): Output report filename
        """
        report_path = os.path.join(self.output_dir, output_file)
        
        print(f"\n📝 Generating detection report...")
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PLANT DISEASE DETECTION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            if not detections:
                f.write("No diseases detected in the video.\n")
            else:
                f.write(f"Total detections: {len(detections)} frames\n\n")
                
                # Group by class
                class_counts = {}
                for det in detections:
                    cls = det['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                f.write("DETECTION SUMMARY:\n")
                f.write("-" * 70 + "\n")
                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(detections)) * 100
                    f.write(f"  • {cls}: {count} frames ({percentage:.1f}%)\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED TIMELINE:\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"{'Frame':<10} {'Time (s)':<12} {'Detection':<40}\n")
                f.write("-" * 70 + "\n")
                
                for det in detections:
                    f.write(f"{det['frame']:<10} {det['time']:<12.2f} {det['class']:<40}\n")
        
        file_size = os.path.getsize(report_path) / 1024
        print(f"✓ Report saved: {report_path} ({file_size:.2f} KB)")
        return report_path


def select_video_file():
    """
    Open file dialog to select a video
    
    Returns:
        file_path: Selected video path or None
    """
    root = Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


def main():
    """
    Standalone execution - Run video processing independently
    """
    print("\n" + "="*60)
    print("VIDEO PROCESSOR - STANDALONE MODE")
    print("="*60 + "\n")
    
    # Import detector
    try:
        from detector import PlantDiseaseDetector
        detector = PlantDiseaseDetector()
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        sys.exit(1)
    
    # Initialize processor
    processor = VideoProcessor(detector)
    
    # Menu
    print("\nSelect mode:")
    print("1. Process video (with file dialog)")
    print("2. Process video (manual path)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Video with dialog
        print("\n📂 Opening file dialog...")
        video_path = select_video_file()
        
        if video_path:
            try:
                snapshots, detections = processor.process_video(video_path)
                
                # Save report
                if detections:
                    processor.save_detection_report(detections)
                else:
                    print("\nℹ️  No detections found - report not generated")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠ No file selected")
    
    elif choice == '2':
        # Video with manual path
        video_path = input("\nEnter video path: ").strip()
        
        if os.path.exists(video_path):
            try:
                snapshots, detections = processor.process_video(video_path)
                
                # Save report
                if detections:
                    processor.save_detection_report(detections)
                else:
                    print("\nℹ️  No detections found - report not generated")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ Error: File not found")
    
    elif choice == '3':
        print("👋 Exiting")
    
    else:
        print("⚠ Invalid choice")


if __name__ == "__main__":
    main()