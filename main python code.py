"""
Main Controller
Central control point for Plant Disease Detection System
Can be run independently to control all modules
"""

import os
import sys
from tkinter import Tk, filedialog
from detector import PlantDiseaseDetector
from image_processor import ImageProcessor
from video_processor import VideoProcessor
import config


class PlantDiseaseDetectionSystem:
    """
    Main controller for the Plant Disease Detection System
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the detection system
        
        Args:
            model_path (str): Path to YOLO model file (uses config if None)
        """
        print("\n" + "="*60)
        print("PLANT DISEASE DETECTION SYSTEM")
        print("="*60)
        
        # Use config values if not provided
        if model_path is None:
            model_path = config.MODEL_PATH
        
        # Initialize detector
        self.detector = PlantDiseaseDetector(
            model_path,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD
        )
        
        # Initialize processors (image and video only)
        self.image_processor = ImageProcessor(self.detector, config.OUTPUT_DIR)
        self.video_processor = VideoProcessor(self.detector, config.OUTPUT_DIR)
        
        # Initialize Tkinter root (hidden) for file dialogs
        self.root = Tk()
        self.root.withdraw()  # Hide the main window
        
        print("✓ System initialized successfully\n")
    
    def select_file(self, file_type="image"):
        """
        Open file dialog to select a file
        
        Args:
            file_type (str): Type of file to select ('image' or 'video')
            
        Returns:
            file_path: Selected file path or None
        """
        if file_type == "image":
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
            title = "Select an Image File"
        elif file_type == "video":
            filetypes = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                ("All files", "*.*")
            ]
            title = "Select a Video File"
        else:
            filetypes = [("All files", "*.*")]
            title = "Select a File"
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        
        return file_path if file_path else None
    
    def select_multiple_files(self, file_type="image"):
        """
        Open file dialog to select multiple files
        
        Args:
            file_type (str): Type of files to select
            
        Returns:
            file_paths: List of selected file paths
        """
        if file_type == "image":
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        else:
            filetypes = [("All files", "*.*")]
        
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Files",
            filetypes=filetypes
        )
        
        return list(file_paths) if file_paths else []
    
    def run_image_detection(self):
        """
        Run detection on a single image
        """
        print("\n📂 Opening file dialog...")
        image_path = self.select_file(file_type="image")
        
        if not image_path:
            print("⚠ No file selected")
            return
        
        try:
            # Process image
            result_path, predicted_class, annotated_image = self.image_processor.process_image(
                image_path, save_result=config.SAVE_IMAGE_RESULTS
            )
            
            # Display result if enabled
            if config.DISPLAY_IMAGE_RESULTS:
                self.image_processor.display_result(annotated_image, config.WINDOW_NAME_IMAGE)
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_batch_image_detection(self):
        """
        Run detection on multiple images
        """
        print("\n📂 Opening file dialog (select multiple images)...")
        image_paths = self.select_multiple_files(file_type="image")
        
        if not image_paths:
            print("⚠ No files selected")
            return
        
        try:
            # Process batch
            results = self.image_processor.batch_process(image_paths)
            
            print("\n" + "="*60)
            print("📊 BATCH RESULTS SUMMARY")
            print("="*60)
            successful = len([r for r in results if r[0]])
            print(f"Total processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print("\nDetailed Results:")
            for idx, (result_path, predicted_class) in enumerate(results, 1):
                status = "✓" if result_path else "✗"
                detection = predicted_class or 'No detection'
                print(f"  {status} {idx}. {detection}")
            print("="*60 + "\n")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_video_detection(self):
        """
        Run detection on a video file
        """
        print("\n📂 Opening file dialog...")
        video_path = self.select_file(file_type="video")
        
        if not video_path:
            print("⚠ No file selected")
            return
        
        try:
            # Process video
            snapshots, detections = self.video_processor.process_video(
                video_path, 
                save_snapshots=config.SAVE_VIDEO_SNAPSHOTS, 
                display_live=config.DISPLAY_LIVE_VIDEO
            )
            
            # Save detection report if enabled
            if detections and config.GENERATE_VIDEO_REPORT:
                self.video_processor.save_detection_report(detections)
            elif not detections:
                print("\nℹ️  No detections found - report not generated")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_menu(self):
        """
        Display the main menu
        """
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Detect from Image (Single)")
        print("2. Detect from Images (Batch)")
        print("3. Detect from Video")
        print("4. View Configuration")
        print("5. Exit")
        print("="*60)
    
    def view_configuration(self):
        """
        Display current configuration
        """
        config.print_config()
    
    def run(self):
        """
        Main execution loop
        """
        while True:
            self.display_menu()
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.run_image_detection()
            
            elif choice == '2':
                self.run_batch_image_detection()
            
            elif choice == '3':
                self.run_video_detection()
            
            elif choice == '4':
                self.view_configuration()
            
            elif choice == '5':
                print("\n" + "="*60)
                print("👋 Exiting system. Goodbye!")
                print("="*60 + "\n")
                break
            
            else:
                print("⚠ Invalid choice. Please select 1-5.")
        
        # Cleanup
        self.root.destroy()


def main():
    """
    Entry point of the application
    """
    try:
        # Create necessary directories
        config.create_directories()
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print("\n⚠ Configuration issues found:")
            for error in errors:
                print(f"  ✗ {error}")
            
            # Continue anyway if only model is missing
            if len(errors) == 1 and "Model file not found" in errors[0]:
                response = input("\nContinue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    sys.exit(1)
            else:
                sys.exit(1)
        
        # Initialize and run the system
        system = PlantDiseaseDetectionSystem()
        system.run()
    
    except KeyboardInterrupt:
        print("\n\n⏸ System interrupted by user")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()