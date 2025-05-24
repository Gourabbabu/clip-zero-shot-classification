# save this as: step2_clip_classification.py

import torch
import clip
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import random

class CLIPClassifier:
    """CLIP-based zero-shot image classifier"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize CLIP classifier"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model {model_name}...")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Initialize class descriptions and embeddings
        self.class_descriptions = {}
        self.text_embeddings = None
        self.class_names = []
        
    def load_class_descriptions_from_file(self, file_path: str = "class_descriptions.py"):
        """Load class descriptions from the generated file"""
        # Read the class_descriptions.py file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Execute the content to get the dictionary
        local_vars = {}
        exec(content, {}, local_vars)
        descriptions = local_vars['class_descriptions']
        
        self.class_descriptions = descriptions
        self.class_names = list(descriptions.keys())
        self._compute_text_embeddings()
        print(f"Loaded {len(self.class_names)} classes from {file_path}")
        return descriptions
        
    def _compute_text_embeddings(self):
        """Compute and cache text embeddings for all class descriptions"""
        if not self.class_descriptions:
            return
            
        text_inputs = [self.class_descriptions[name] for name in self.class_names]
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(text_tokens)
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
    
    def predict_image(self, image_path: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Predict class for a single image"""
        if self.text_embeddings is None:
            raise ValueError("No class descriptions loaded")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarities = (image_embedding @ self.text_embeddings.T).softmax(dim=-1)
            
        # Get top-k predictions
        top_similarities, top_indices = similarities[0].topk(min(top_k, len(self.class_names)))
        
        top_classes = [self.class_names[idx] for idx in top_indices.cpu().numpy()]
        top_scores = top_similarities.cpu().numpy().tolist()
        
        return top_classes, top_scores
    
    def add_new_classes(self, new_descriptions: Dict[str, str]):
        """Add new classes dynamically"""
        print(f"Adding {len(new_descriptions)} new classes...")
        self.class_descriptions.update(new_descriptions)
        self.class_names = list(self.class_descriptions.keys())
        self._compute_text_embeddings()
        print(f"Total classes now: {len(self.class_names)}")

def load_test_images(dataset_path: str = "./dataset", images_per_class: int = 10):
    """Load test images from your dataset"""
    test_images = []
    test_labels = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            # Get all images in this class folder
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Select random images for testing
            selected_images = random.sample(image_files, min(images_per_class, len(image_files)))
            
            for img_file in selected_images:
                img_path = os.path.join(class_path, img_file)
                test_images.append(img_path)
                test_labels.append(class_folder)
    
    print(f"Loaded {len(test_images)} test images from {len(set(test_labels))} classes")
    return test_images, test_labels

def evaluate_classifier(classifier, test_images, test_labels):
    """Evaluate the classifier on test images"""
    print("Evaluating classifier...")
    
    all_predictions = []
    all_top5_predictions = []
    
    for img_path in tqdm(test_images, desc="Processing test images"):
        try:
            top_classes, top_scores = classifier.predict_image(img_path, top_k=5)
            all_predictions.append(top_classes[0] if top_classes else "")
            all_top5_predictions.append(top_classes)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            all_predictions.append("")
            all_top5_predictions.append([])
    
    # Calculate Top-1 accuracy
    top1_accuracy = accuracy_score(test_labels, all_predictions)
    
    # Calculate Top-5 accuracy
    top5_correct = 0
    for true_label, top5_preds in zip(test_labels, all_top5_predictions):
        if true_label in top5_preds:
            top5_correct += 1
    top5_accuracy = top5_correct / len(test_labels)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(test_labels, all_predictions, zero_division=0))
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'predictions': all_predictions,
        'true_labels': test_labels,
        'top5_predictions': all_top5_predictions
    }

def plot_confusion_matrix(results, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    true_labels = results['true_labels']
    pred_labels = results['predictions']
    
    # Get unique labels and create confusion matrix
    unique_labels = sorted(list(set(true_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # Plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('CLIP Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved as {save_path}")

def demo_single_prediction(classifier, test_images, test_labels):
    """Demo single image prediction"""
    print(f"\n=== SINGLE IMAGE PREDICTION DEMO ===")
    
    # Pick a random test image
    idx = random.randint(0, len(test_images)-1)
    sample_image = test_images[idx]
    true_label = test_labels[idx]
    
    print(f"Image: {sample_image}")
    print(f"True label: {true_label}")
    
    # Get predictions
    classes, scores = classifier.predict_image(sample_image, top_k=5)
    
    print("Top-5 predictions:")
    for i, (cls, score) in enumerate(zip(classes, scores), 1):
        marker = "✓" if cls == true_label else " "
        print(f"{marker} {i}. {cls}: {score:.4f} ({score*100:.2f}%)")

def demo_class_expansion(classifier):
    """Demo adding new classes"""
    print(f"\n=== DYNAMIC CLASS EXPANSION DEMO ===")
    print(f"Current number of classes: {len(classifier.class_names)}")
    
    # Add some new classes
    new_classes = {
        "helicopter": "a photo of a helicopter",
        "submarine": "a photo of a submarine", 
        "telescope": "a photo of a telescope"
    }
    
    classifier.add_new_classes(new_classes)
    print(f"New classes added: {list(new_classes.keys())}")
    print(f"Total classes now: {len(classifier.class_names)}")

if __name__ == "__main__":
    print("🚀 Starting CLIP Zero-Shot Image Classification")
    print("=" * 50)
    
    # Step 1: Initialize CLIP classifier
    classifier = CLIPClassifier(model_name="ViT-B/32")
    
    # Step 2: Load class descriptions from your generated file
    class_descriptions = classifier.load_class_descriptions_from_file("class_descriptions.py")
    print(f"Classes loaded: {list(class_descriptions.keys())[:5]}...")  # Show first 5
    
    # Step 3: Load test images from your dataset
    test_images, test_labels = load_test_images("./dataset", images_per_class=10)
    
    # Step 4: Evaluate the classifier
    results = evaluate_classifier(classifier, test_images, test_labels)
    
    # Step 5: Plot confusion matrix
    plot_confusion_matrix(results)
    
    # Step 6: Demo single prediction
    demo_single_prediction(classifier, test_images, test_labels)
    
    # Step 7: Demo class expansion
    demo_class_expansion(classifier)
    
    # Step 8: Save results
    with open('evaluation_results.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {
            'top1_accuracy': results['top1_accuracy'],
            'top5_accuracy': results['top5_accuracy'],
            'num_test_images': len(test_images),
            'num_classes': len(set(test_labels))
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Complete! Results saved to evaluation_results.json")
    print(f"📊 Confusion matrix saved as confusion_matrix.png")