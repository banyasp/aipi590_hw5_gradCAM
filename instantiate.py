"""
YOLOv11 Inference and GradCAM Visualization on Emotion Dataset
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import random
from typing import List, Tuple

class YOLOGradCAM:
    """
    GradCAM implementation for YOLO models
    """
    def __init__(self, model, target_layers=None):
        """
        Initialize GradCAM
        
        Parameters:
        -----------
        model : YOLO
            The YOLO model
        target_layers : list, optional
            List of target layers for GradCAM
        """
        self.model = model
        self.gradients = []
        self.activations = []
        
        # Find suitable layers if not provided
        if target_layers is None:
            target_layers = self._find_target_layers()
        
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.handlers = []
        
        # Register hooks for each target layer
        for layer in self.target_layers:
            self.handlers.append(
                layer.register_forward_hook(self.save_activation)
            )
    
    def _find_target_layers(self):
        """Find suitable convolutional layers in the model"""
        target_layers = []
        for name, module in self.model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layers.append(module)
        # Return the last few conv layers
        return target_layers[-3:] if len(target_layers) >= 3 else target_layers
    
    def save_activation(self, module, input, output):
        """Save the activations during forward pass"""
        self.activations.append(output.cpu().detach())
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handler in self.handlers:
            handler.remove()
    
    def generate_cam(self, input_image, target_category=None):
        """
        Generate GradCAM heatmap
        
        Parameters:
        -----------
        input_image : numpy.ndarray
            Input image
        target_category : int, optional
            Target class index for GradCAM
        
        Returns:
        --------
        cam : numpy.ndarray
            GradCAM heatmap
        """
        from PIL import Image
        
        # Reset activations
        self.activations = []
        
        # Prepare input tensor
        if isinstance(input_image, np.ndarray):
            img_pil = Image.fromarray(input_image)
        else:
            img_pil = input_image
        
        # Preprocess image similar to YOLO preprocessing
        img_resized = img_pil.resize((640, 640))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        # Move to the same device as the model
        device = next(self.model.model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Forward pass to capture activations
        with torch.no_grad():
            # Run inference
            results = self.model.predict(input_image, verbose=False)
        
        # Process activations to create heatmap
        if len(self.activations) > 0:
            # Use the last activation layer
            activation = self.activations[-1]
            
            # Average across channels to get a single heatmap
            if len(activation.shape) == 4:  # [batch, channels, height, width]
                heatmap = torch.mean(activation[0], dim=0).numpy()
            else:
                heatmap = activation[0].numpy()
            
            # Apply ReLU
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize the heatmap
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            return heatmap, results[0] if len(results) > 0 else None
        
        # Return empty heatmap if no activations
        return np.zeros((20, 20)), results[0] if len(results) > 0 else None
    

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image
    heatmap : numpy.ndarray
        GradCAM heatmap
    alpha : float
        Transparency of the heatmap
    colormap : int
        OpenCV colormap
    
    Returns:
    --------
    superimposed_img : numpy.ndarray
        Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    # Convert BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img


def draw_detections(img, result):
    """
    Draw bounding boxes and labels on image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image
    result : ultralytics.engine.results.Results
        YOLO prediction results
    
    Returns:
    --------
    img_with_boxes : numpy.ndarray
        Image with bounding boxes
    """
    img_copy = img.copy()
    
    if result is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{result.names[int(cls)]}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_copy


def get_sample_images(emotion_dataset_path: str, num_samples: int = 5) -> List[Tuple[str, str]]:
    """
    Get sample images from each emotion category
    
    Parameters:
    -----------
    emotion_dataset_path : str
        Path to emotion dataset
    num_samples : int
        Number of samples per emotion
    
    Returns:
    --------
    samples : list
        List of (image_path, emotion) tuples
    """
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    samples = []
    
    for emotion in emotions:
        emotion_path = os.path.join(emotion_dataset_path, emotion)
        if os.path.exists(emotion_path):
            images = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Get random samples
            selected = random.sample(images, min(num_samples, len(images)))
            for img_name in selected:
                samples.append((os.path.join(emotion_path, img_name), emotion))
    
    return samples


def run_inference_and_gradcam(model_path: str, 
                              emotion_dataset_path: str,
                              output_dir: str = 'gradcam_results',
                              num_samples: int = 3):
    """
    Run YOLO inference and GradCAM visualization
    
    Parameters:
    -----------
    model_path : str
        Path to YOLO model
    emotion_dataset_path : str
        Path to emotion dataset
    output_dir : str
        Directory to save results
    num_samples : int
        Number of samples per emotion to process
    """
    print("="*60)
    print("YOLO Inference and GradCAM Visualization")
    print("="*60)
    
    # Load YOLO model
    print(f"\n[1/4] Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully!")
    print(f"   Model type: {type(model.model).__name__}")
    
    # Initialize GradCAM (it will automatically find suitable layers)
    print(f"\n[2/4] Initializing GradCAM...")
    gradcam = YOLOGradCAM(model)
    print(f"✓ GradCAM initialized with {len(gradcam.target_layers)} target layers!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {os.path.abspath(output_dir)}")
    
    # Get sample images
    print(f"\n[3/4] Loading sample images from emotion dataset...")
    samples = get_sample_images(emotion_dataset_path, num_samples)
    print(f"✓ Loaded {len(samples)} sample images")
    
    # Process each image
    print(f"\n[4/4] Processing images and generating GradCAM visualizations...")
    print("-"*60)
    
    results_data = []
    
    for idx, (img_path, emotion) in enumerate(samples, 1):
        print(f"\n[{idx}/{len(samples)}] Processing: {os.path.basename(img_path)} (Emotion: {emotion})")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ✗ Failed to load image")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        print(f"  → Running YOLO inference...")
        results = model.predict(img_path, verbose=False)
        
        # Get detections info
        num_detections = len(results[0].boxes) if results else 0
        print(f"  ✓ Detections found: {num_detections}")
        
        if num_detections > 0:
            boxes = results[0].boxes
            for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                class_name = results[0].names[int(cls)]
                print(f"    - Object {i+1}: {class_name} (confidence: {conf:.3f})")
        
        # Generate GradCAM
        print(f"  → Generating GradCAM heatmap...")
        heatmap, result = gradcam.generate_cam(img_rgb)
        print(f"  ✓ GradCAM heatmap generated!")
        
        # Create visualization
        img_with_detections = draw_detections(img_rgb, result)
        img_with_heatmap = overlay_heatmap(img_rgb, heatmap)
        
        # Create combined visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Emotion: {emotion.upper()} | File: {os.path.basename(img_path)}', 
                     fontsize=14, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Image with detections
        axes[0, 1].imshow(img_with_detections)
        axes[0, 1].set_title(f'YOLO Detections ({num_detections} objects)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # GradCAM heatmap
        axes[1, 0].imshow(heatmap, cmap='jet')
        axes[1, 0].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Image with GradCAM overlay
        axes[1, 1].imshow(img_with_heatmap)
        axes[1, 1].set_title('GradCAM Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_filename = f"{emotion}_{idx:03d}_gradcam.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization to: {output_filename}")
        
        # Store results
        results_data.append({
            'emotion': emotion,
            'image': os.path.basename(img_path),
            'num_detections': num_detections,
            'output_file': output_filename
        })
    
    # Create summary visualization
    print(f"\n[5/5] Creating summary visualization...")
    create_summary_report(results_data, output_dir)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\n✓ Total images processed: {len(results_data)}")
    print(f"✓ Output directory: {os.path.abspath(output_dir)}")
    print(f"✓ All visualizations saved successfully!")
    print("\n" + "="*60)


def create_summary_report(results_data, output_dir):
    """
    Create a summary report of all processed images
    
    Parameters:
    -----------
    results_data : list
        List of dictionaries containing results information
    output_dir : str
        Directory to save the report
    """
    # Create summary statistics
    from collections import Counter
    
    emotion_counts = Counter([r['emotion'] for r in results_data])
    total_detections = sum([r['num_detections'] for r in results_data])
    
    # Create summary text file
    summary_path = os.path.join(output_dir, 'SUMMARY_REPORT.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YOLO INFERENCE AND GRADCAM VISUALIZATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Images Processed: {len(results_data)}\n")
        f.write(f"Total Objects Detected: {total_detections}\n\n")
        
        f.write("Images per Emotion:\n")
        for emotion, count in sorted(emotion_counts.items()):
            f.write(f"  - {emotion.capitalize()}: {count}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Detailed Results:\n")
        f.write("-"*60 + "\n\n")
        
        for idx, result in enumerate(results_data, 1):
            f.write(f"{idx}. Image: {result['image']}\n")
            f.write(f"   Emotion: {result['emotion'].capitalize()}\n")
            f.write(f"   Detections: {result['num_detections']}\n")
            f.write(f"   Output: {result['output_file']}\n\n")
    
    print(f"✓ Summary report saved to: SUMMARY_REPORT.txt")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/Users/peterbanyas/Desktop/AIPI 590/HW5/models/yolo11n.pt"
    EMOTION_DATASET_PATH = "/Users/peterbanyas/Desktop/AIPI 590/HW5/emotion_dataset"
    OUTPUT_DIR = "/Users/peterbanyas/Desktop/AIPI 590/HW5/gradcam_results"
    NUM_SAMPLES_PER_EMOTION = 2  # Number of samples per emotion category
    
    # Run the inference and GradCAM visualization
    run_inference_and_gradcam(
        model_path=MODEL_PATH,
        emotion_dataset_path=EMOTION_DATASET_PATH,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES_PER_EMOTION
    )

