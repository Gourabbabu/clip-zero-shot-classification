# save this as: step3_quick_test.py

import torch
import clip
from PIL import Image
import os
import random

def quick_test():
    """Quick test to make sure everything works"""
    print("🧪 Quick Test - CLIP Classification")
    print("=" * 40)
    
    # Check if required files exist
    if not os.path.exists("class_descriptions.py"):
        print("❌ class_descriptions.py not found!")
        return False
        
    if not os.path.exists("./dataset"):
        print("❌ dataset folder not found!")
        return False
    
    # Load CLIP
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ CLIP loaded on {device}")
    
    # Load class descriptions
    with open("class_descriptions.py", 'r') as f:
        content = f.read()
    local_vars = {}
    exec(content, {}, local_vars)
    class_descriptions = local_vars['class_descriptions']
    
    print(f"✅ Loaded {len(class_descriptions)} classes")
    print(f"Sample classes: {list(class_descriptions.keys())[:5]}")
    
    # Find a random image
    dataset_folders = [d for d in os.listdir("./dataset") if os.path.isdir(os.path.join("./dataset", d))]
    if not dataset_folders:
        print("❌ No class folders found in dataset!")
        return False
        
    # Pick random class and image
    random_class = random.choice(dataset_folders)
    class_path = os.path.join("./dataset", random_class)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {random_class} folder!")
        return False
        
    random_image = random.choice(image_files)
    image_path = os.path.join(class_path, random_image)
    
    print(f"Testing with: {image_path}")
    print(f"True class: {random_class}")
    
    # Test prediction
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Prepare text inputs
        text_inputs = list(class_descriptions.values())
        text_tokens = clip.tokenize(text_inputs).to(device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            
        # Get top predictions
        class_names = list(class_descriptions.keys())
        top_similarities, top_indices = similarities[0].topk(5)
        
        print("\n🎯 Top 5 Predictions:")
        for i, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
            class_name = class_names[idx]
            confidence = sim.item()
            marker = "✅" if class_name == random_class else "  "
            print(f"{marker} {i+1}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
            
        print(f"\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Everything looks good! You can now run the main script.")
        print("Next step: python step2_clip_classification.py")
    else:
        print("\n❌ There are some issues. Please check the error messages above.")