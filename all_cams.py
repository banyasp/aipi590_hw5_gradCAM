# bult with claude sonnet 4.5, Oct 6 @ 8:40am
# Optimized for faster ScoreCAM processing with parallel image processing

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import ssl
import certifi
import os
from multiprocessing import Pool, cpu_count
import time

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def process_single_image(img_info):
    """
    Worker function to process a single image with all CAM methods.
    This function will be called in parallel for each image.
    """
    img_idx, total_imgs, emotion_folder, img_name = img_info
    
    # Each worker initializes its own model and CAM instances
    # Note: For GPU, this will use the same GPU but different CUDA contexts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model for this worker
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device)
    model.eval()
    
    # Target layer (last conv layer in ResNet50)
    target_layers = [model.layer4[-1]]
    
    # Initialize CAM methods for this worker
    cam_algorithms = {
        'GradCAM': GradCAM(model=model, target_layers=target_layers),
        'GradCAM++': GradCAMPlusPlus(model=model, target_layers=target_layers),
        'EigenCAM': EigenCAM(model=model, target_layers=target_layers),
        'Score-CAM': ScoreCAM(model=model, target_layers=target_layers)
    }
    
    print(f"\n{'='*60}")
    print(f"Processing image {img_idx}/{total_imgs}: {img_name}")
    print(f"{'='*60}")
    
    # Load and preprocess image
    img = Image.open(f'emotion_dataset/{emotion_folder}/{img_name}.jpg').convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)
    
    # Generate visualizations for all methods
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Original image
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Generate CAM for each method
    for idx, (name, cam) in enumerate(cam_algorithms.items(), 1):
        print(f"  [{img_name}] Generating {name}...")
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)
        visualization = show_cam_on_image(img_array, grayscale_cam[0], use_rgb=True)
        
        axes[idx].imshow(visualization)
        axes[idx].set_title(name)
        axes[idx].axis('off')
        
        # Save individual visualizations
        output_img = Image.fromarray(visualization)
        output_img.save(f'output/{name.lower().replace("-", "")}_output_{img_name}.jpg')
        print(f"    [{img_name}] Saved output/{name.lower().replace('-', '')}_output_{img_name}.jpg")
    
    plt.tight_layout()
    comparison_path = f'output/cam_comparison_{img_name}.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  [{img_name}] Comparison saved as '{comparison_path}'")
    plt.close()
    
    # Print predicted class
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()
        print(f"  [{img_name}] Predicted class index: {pred_class}")
        print(f"  [{img_name}] Confidence: {confidence:.2%}")
    
    # Clean up
    del model, cam_algorithms
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return img_name

if __name__ == '__main__':
    # List of images to process - add your image paths here
    # Format: (emotion_folder, image_name)
    img_list = [
        # ('happy', 'happy_01932'),
        # ('happy', 'happy_01419'),
        # ('disgust', 'disgust_00024'),
        # ('disgust', 'disgust_00045'),
        # ('angry', 'angry_00009'),
        # ('angry', 'angry_00021'),
        # ('surprise', 'surprise_00007'),
        ('surprise', 'surprise_00019'),
        # ('sad', 'sad_00007'),
        # ('sad', 'sad_00018'),
        # ('neutral', 'neutral_00017'),
        # ('neutral', 'neutral_00025')
    ]
    
    # Configure number of parallel workers
    # For CPU: use cpu_count() or cpu_count() - 1
    # For GPU: use 2-4 to avoid GPU memory issues (GPU context switching has overhead)
    num_workers = cpu_count() - 1
    # min(4, len(img_list))  # Use up to 4 workers, but no more than number of images
    
    print(f"\n{'='*60}")
    print(f"Starting parallel processing with {num_workers} workers")
    print(f"Total images to process: {len(img_list)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Prepare arguments for each worker (add index and total count)
    img_info_list = [
        (idx, len(img_list), emotion_folder, img_name)
        for idx, (emotion_folder, img_name) in enumerate(img_list, 1)
    ]
    
    # Process images in parallel using multiprocessing
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image, img_info_list)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"All {len(img_list)} images processed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(img_list):.2f} seconds")
    print(f"{'='*60}")