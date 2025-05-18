import os
import sys
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, MobileViTV2ForSemanticSegmentation

def visualize_segmentation(logits, original_image, output_path):
    """
    Visualize semantic segmentation results.
    
    Args:
        logits: Model output logits of shape (batch_size, num_classes, height, width)
        original_image: Original PIL image
        output_path: Path to save the visualization
    """
    # Get predictions by taking argmax along the class dimension
    predictions = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Create a color map (one color per class)
    num_classes = logits.shape[1]
    # Generate a color map with distinct colors
    cmap = plt.cm.get_cmap('viridis', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
    
    # Create a colorized segmentation map
    segmentation_map = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        mask = predictions == class_idx
        for c in range(3):  # RGB channels
            segmentation_map[:, :, c][mask] = int(colors[class_idx][c] * 255)
    
    # Convert to PIL Image
    segmentation_image = Image.fromarray(segmentation_map)
    
    # Resize original image to match segmentation map size
    original_image_resized = original_image.resize(
        (segmentation_map.shape[1], segmentation_map.shape[0]), 
        Image.LANCZOS
    )
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot segmentation map
    axes[1].imshow(segmentation_map)
    axes[1].set_title('Segmentation Map')
    axes[1].axis('off')
    
    # Create a blended overlay (50% original, 50% segmentation)
    overlay = Image.blend(
        original_image_resized.convert('RGBA'), 
        segmentation_image.convert('RGBA'), 
        alpha=0.5
    )
    axes[2].imshow(np.array(overlay))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return segmentation_map

def process_image(image_url=None, image_path=None, output_dir="output"):
    """
    Process an image for semantic segmentation.
    
    Args:
        image_url: URL of the image to process (optional)
        image_path: Path to a local image file (optional)
        output_dir: Directory to save the output (default: "output")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image from URL or local path
    if image_url:
        print(f"Loading image from URL: {image_url}")
        image = Image.open(requests.get(image_url, stream=True).raw)
    elif image_path:
        print(f"Loading image from path: {image_path}")
        image = Image.open(image_path)
    else:
        # Default image if none provided
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        print(f"Using default image from URL: {url}")
        image = Image.open(requests.get(url, stream=True).raw)
    
    # Save the original image
    image_filename = os.path.join(output_dir, "input_image.jpg")
    image.save(image_filename)
    print(f"Saved input image to {image_filename}")
    
    # Load model and processor
    print("Loading model and processor...")
    image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
    model = MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
    
    # Process image
    print("Processing image...")
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits
    logits = outputs.logits
    print(f"Segmentation complete. Logits shape: {logits.shape}")
    
    # Save the logits as a tensor
    torch.save(logits, os.path.join(output_dir, "segmentation_logits.pt"))
    print(f"Saved logits to {os.path.join(output_dir, 'segmentation_logits.pt')}")
    
    # Visualize the segmentation results
    print("Generating visualization...")
    vis_path = os.path.join(output_dir, "segmentation_visualization.png")
    visualize_segmentation(logits, image, vis_path)
    print(f"Saved visualization to {vis_path}")
    
    return logits

if __name__ == "__main__":
    # Parse command-line arguments
    image_url = None
    image_path = None
    output_dir = "output"
    
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("http"):
            image_url = sys.argv[1]
        else:
            image_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Process the image
    process_image(image_url, image_path, output_dir)
    print("Processing complete!")
