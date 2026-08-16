"""
Configuration File
Central configuration for the Plant Disease Detection System
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = "model/best.onnx"
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence for detection (0.0 to 1.0)
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for NMS

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_DIR = "uploaded_datasets"  # Directory for saving results
ENABLE_AUTO_SAVE = True  # Automatically save detection results

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
DEFAULT_CAMERA_ID = 0  # Default camera device ID
ALERT_DURATION = 2.0  # Seconds of continuous detection before alert

# ============================================================================
# VIDEO PROCESSING CONFIGURATION
# ============================================================================
DISPLAY_LIVE_VIDEO = True  # Show live video during processing
SAVE_VIDEO_SNAPSHOTS = True  # Save frames with detections
GENERATE_VIDEO_REPORT = True  # Generate text report of detections

# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================
DISPLAY_IMAGE_RESULTS = True  # Show detection results in window
SAVE_IMAGE_RESULTS = True  # Save annotated images

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================
WINDOW_NAME_IMAGE = "Plant Disease Detection - Image"
WINDOW_NAME_VIDEO = "Plant Disease Detection - Video"
WINDOW_NAME_CAMERA = "Plant Disease Detection - Live Camera"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
ENABLE_LOGGING = True
LOG_FILE = "detection_log.txt"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

 

# ============================================================================
# FILE EXTENSIONS
# ============================================================================
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """
    Create necessary directories if they don't exist
    """
    directories = [OUTPUT_DIR, "logs", "reports"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def validate_config():
    """
    Validate configuration settings
    
    Returns:
        errors (list): List of configuration errors
    """
    errors = []
    
    # Check model path
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {MODEL_PATH}")
    
    # Validate thresholds
    if not (0.0 <= CONFIDENCE_THRESHOLD <= 1.0):
        errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    if not (0.0 <= IOU_THRESHOLD <= 1.0):
        errors.append("IOU_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate alert duration
    if ALERT_DURATION < 0:
        errors.append("ALERT_DURATION must be positive")
    
    return errors

def get_config_summary():
    """
    Get a summary of current configuration
    
    Returns:
        summary (dict): Configuration summary
    """
    return {
        'model_path': MODEL_PATH,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'output_dir': OUTPUT_DIR,
        'camera_id': DEFAULT_CAMERA_ID,
        'alert_duration': ALERT_DURATION,
    }

def print_config():
    """
    Print current configuration to console
    """
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IOU Threshold: {IOU_THRESHOLD}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Default Camera ID: {DEFAULT_CAMERA_ID}")
    print(f"Alert Duration: {ALERT_DURATION}s")
    print("="*60 + "\n")

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONFIGURATION VALIDATION")
    print("="*60 + "\n")
    
    # Display current config
    print_config()
    
    # Create directories
    print("Creating directories...")
    create_directories()
    
    # Validate configuration
    print("\nValidating configuration...")
    errors = validate_config()
    
    if errors:
        print("\n⚠ Configuration errors found:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ Configuration is valid")
        print("✓ All checks passed")
    
    print("\n" + "="*60)