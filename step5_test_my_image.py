# step5_test_my_image.py
import clip
import torch
from PIL import Image
import numpy as np

def classify_custom_image():
    print("🖼️ Loading your custom image for classification...")
    
    # Your image path
    image_path = r"C:\Users\gourab\Downloads\premium_photo-1689551670902-19b441a6afde.jpg"
    
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    try:
        # Load and preprocess your image
        image = Image.open(image_path)
        print(f"✅ Successfully loaded image: {image.size}")
        
        # Show basic image info
        print(f"Image mode: {image.mode}")
        print(f"Image format: {image.format}")
        
        # Preprocess for CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Define classes to test against
        # You can modify these based on what you think your image might be
        test_classes = [
            "a photo of a dog",
            "a photo of a cat", 
            "a photo of a car",
            "a photo of a person",
            "a photo of a house",
            "a photo of a tree",
            "a photo of food",
            "a photo of a flower",
            "a photo of an animal",
            "a photo of a landscape",
            "a photo of a building",
            "a photo of technology",
            "a photo of furniture",
            "a photo of clothing",
            "a photo of a vehicle"
        ]
        
        # Encode text prompts
        text_inputs = clip.tokenize(test_classes).to(device)
        
        # Get predictions
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Calculate similarities
            logits_per_image, logits_per_text = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Show top 5 predictions
        print("\n🎯 Top 5 Predictions:")
        print("-" * 50)
        
        # Get top 5 indices
        top5_idx = np.argsort(probs[0])[::-1][:5]
        
        for i, idx in enumerate(top5_idx):
            confidence = probs[0][idx] * 100
            print(f"{i+1}. {test_classes[idx]:<30} ({confidence:.2f}%)")
        
        # Show the most likely prediction
        best_match_idx = top5_idx[0]
        best_confidence = probs[0][best_match_idx] * 100
        
        print(f"\n🏆 Best Match: {test_classes[best_match_idx]}")
        print(f"Confidence: {best_confidence:.2f}%")
        
        # Optional: Display the image (if you have matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Prediction: {test_classes[best_match_idx]}\nConfidence: {best_confidence:.2f}%")
            plt.tight_layout()
            plt.show()
            print("\n📊 Image displayed with prediction!")
        except ImportError:
            print("\n💡 Install matplotlib to see the image: pip install matplotlib")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find image at {image_path}")
        print("Please check if the file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")

if __name__ == "__main__":
    classify_custom_image()