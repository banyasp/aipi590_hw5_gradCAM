# bult with claude sonnet 4.5, Oct 6 @ 8:40am

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
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

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Load pretrained model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Target layer (last conv layer in ResNet50)
target_layers = [model.layer4[-1]]

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

img_name = 'happy_01932'
# Load and preprocess an image
img = Image.open(f'emotion_dataset/happy/{img_name}.jpg').convert('RGB')
img_resized = img.resize((224, 224))
img_array = np.array(img_resized) / 255.0
img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)

# Initialize all three CAM methods
cam_algorithms = {
    'GradCAM': GradCAM(model=model, target_layers=target_layers),
    'GradCAM++': GradCAMPlusPlus(model=model, target_layers=target_layers),
    'Score-CAM': ScoreCAM(model=model, target_layers=target_layers)
}

# Generate visualizations for all methods
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original image
axes[0].imshow(img_resized)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Generate CAM for each method
for idx, (name, cam) in enumerate(cam_algorithms.items(), 1):
    print(f"Generating {name}...")
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    visualization = show_cam_on_image(img_array, grayscale_cam[0], use_rgb=True)
    
    axes[idx].imshow(visualization)
    axes[idx].set_title(name)
    axes[idx].axis('off')
    
    # Save individual visualizations
    output_img = Image.fromarray(visualization)
    output_img.save(f'output/{name.lower().replace("-", "")}_output_{img_name}.jpg')
    print(f"  - Saved output/{name.lower().replace('-', '')}_output_{img_name}.jpg")

plt.tight_layout()
comparison_path = f'output/cam_comparison_{img_name}.png'
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"\nComparison saved as '{comparison_path}'")
plt.close()

# Print predicted class
with torch.no_grad():
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    print(f"\nPredicted class index: {pred_class}")
    print(f"Confidence: {confidence:.2%}")