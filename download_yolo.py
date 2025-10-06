"""
Download a pretrained YOLOv8 model from Ultralytics
"""

from ultralytics import YOLO
import os

def download_yolov8_model(model_size='n', save_dir='models'):
    """
    Download a pretrained YOLOv8 model
    
    Parameters:
    -----------
    model_size : str
        Size of the model to download. Options:
        - 'n' : nano (fastest, least accurate)
        - 's' : small
        - 'm' : medium
        - 'l' : large
        - 'x' : xlarge (slowest, most accurate)
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    model : YOLO
        The loaded YOLOv8 model
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Model name
    model_name = f'yolov8{model_size}.pt'
    model_path = os.path.join(save_dir, model_name)
    
    print(f"Downloading YOLOv8{model_size.upper()} model...")
    
    # Load the model (this will automatically download if not present)
    model = YOLO(model_name)
    
    print(f"✓ Model downloaded successfully!")
    print(f"✓ Model saved to: {os.path.abspath(model_path)}")
    print(f"\nModel info:")
    print(f"  - Name: YOLOv8{model_size.upper()}")
    print(f"  - Task: Object Detection")
    print(f"  - Input size: 640x640")
    print(f"  - Number of classes: 80 (COCO dataset)")
    
    return model


if __name__ == "__main__":
    # Download YOLOv8 nano model (smallest and fastest)
    # You can change 'n' to 's', 'm', 'l', or 'x' for larger models
    model = download_yolov8_model(model_size='n', save_dir='models')
    
    # Optional: Print model summary
    print("\n" + "="*50)
    print("Model Summary:")
    print("="*50)
    model.info()
    
    # Optional: Test the model with a simple prediction
    print("\n" + "="*50)
    print("Model is ready for inference!")
    print("="*50)
    print("\nExample usage:")
    print("  results = model('path/to/image.jpg')")
    print("  results = model.predict('path/to/image.jpg', save=True)")

