import os
import shutil
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def prepare_cifar100(output_dir="./dataset"):
    """
    Download CIFAR-100 dataset and organize it into folders by class.
    
    Args:
        output_dir: Directory to save the organized dataset
    
    Returns:
        class_descriptions: Dictionary mapping class names to descriptions
    """
    # Download CIFAR-100 dataset
    print("Downloading CIFAR-100 dataset...")
    transform = transforms.ToTensor()
    dataset = CIFAR100(root='./data', download=True, transform=transform)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    class_names = dataset.classes
    
    # Create class descriptions dictionary
    class_descriptions = {class_name: f"a photo of a {class_name}" for class_name in class_names}
    
    # Create directories for each class and save images
    for i, (image, label) in enumerate(dataset):
        class_name = class_names[label]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Convert tensor to PIL Image and save
        img = transforms.ToPILImage()(image)
        img.save(os.path.join(class_dir, f"{i}.jpg"))
        
        # Print progress
        if i % 500 == 0:
            print(f"Processed {i}/{len(dataset)} images")
    
    print(f"Dataset organized into {output_dir}")
    return class_descriptions

def prepare_custom_dataset(input_dir, output_dir="./dataset"):
    """
    Organize a custom dataset into a structured format.
    Assumes input_dir has subdirectories for each class.
    
    Args:
        input_dir: Directory containing class subdirectories with images
        output_dir: Directory to save the organized dataset
    
    Returns:
        class_descriptions: Dictionary mapping class names to descriptions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names (folder names)
    class_names = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    
    # Create class descriptions dictionary
    class_descriptions = {class_name: f"a photo of a {class_name}" for class_name in class_names}
    
    # Copy directories for each class
    for class_name in class_names:
        src_dir = os.path.join(input_dir, class_name)
        dst_dir = os.path.join(output_dir, class_name)
        
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        
        shutil.copytree(src_dir, dst_dir)
        print(f"Copied class: {class_name}")
    
    print(f"Custom dataset organized into {output_dir}")
    return class_descriptions

def visualize_dataset(dataset_dir, num_samples=5):
    """
    Visualize random samples from the dataset
    
    Args:
        dataset_dir: Directory containing the organized dataset
        num_samples: Number of samples to visualize per class
    """
    # Get class names (folder names)
    class_names = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, d))]
    
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(12, 2*len(class_names)))
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random samples
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for j, sample in enumerate(samples):
            img_path = os.path.join(class_dir, sample)
            img = Image.open(img_path)
            
            if len(class_names) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
                
            ax.imshow(img)
            ax.set_title(class_name if j == 0 else "")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_class_descriptions(class_descriptions, output_file="class_descriptions.py"):
    """
    Save class descriptions dictionary to a Python file
    
    Args:
        class_descriptions: Dictionary mapping class names to descriptions
        output_file: File path to save the dictionary
    """
    with open(output_file, 'w') as f:
        f.write("class_descriptions = {\n")
        for class_name, description in class_descriptions.items():
            f.write(f'    "{class_name}": "{description}",\n')
        f.write("}\n")
    
    print(f"Class descriptions saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    
    # Option 1: Prepare CIFAR-100 dataset
    class_descriptions = prepare_cifar100(output_dir="./dataset")
    
    # Option 2: Prepare custom dataset
    # class_descriptions = prepare_custom_dataset(input_dir="./my_images", output_dir="./dataset")
    
    # Save class descriptions
    save_class_descriptions(class_descriptions)
    
    # Visualize the dataset
    visualize_dataset("./dataset", num_samples=5)
    
    # Print dataset structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk("./dataset"):
        level = root.replace("./dataset", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level == 1:  # Show only a few images per class
            for f in files[:3]:
                print(f"{indent}    {f}")
            if len(files) > 3:
                print(f"{indent}    ...")