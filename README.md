# Zero-Shot Image Classification System Using CLIP

A comprehensive zero-shot image classification system built with OpenAI's CLIP (Contrastive Language-Image Pre-Training) model. This system can classify images into any categories without requiring training data.

## 🌟 Features

- **Zero-shot classification** - No training required for new categories
- **Multi-model support** - Works with different CLIP model variants
- **Batch processing** - Classify multiple images efficiently
- **Custom categories** - Define your own classification categories
- **Performance metrics** - Comprehensive accuracy analysis
- **Visualization** - Confusion matrices and result plots
- **Easy to use** - Simple scripts for quick testing

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Zero-Shot-Image-Classification-CLIP.git
cd Zero-Shot-Image-Classification-CLIP
pip install -r requirements.txt
```

### Basic Usage

1. **Quick Test**
```bash
python src/quick_test.py
```

2. **Classify Your Own Image**
```bash
python src/custom_image_test.py
```

3. **Full CIFAR-100 Evaluation**
```bash
python src/clip_classification.py
```

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── clip_classification.py    # Main classification system
│   ├── quick_test.py            # Quick functionality test
│   ├── custom_image_test.py     # Single image classification
│   └── class_descriptions.py    # CIFAR-100 class definitions
├── data/                        # Data directory
│   └── sample_images/           # Sample test images
├── results/                     # Output results and plots
├── docs/                        # Documentation
└── requirements.txt             # Python dependencies
```

## 🔧 Usage Examples

### Classify Single Image

```python
from src.custom_image_test import classify_custom_image

# Classify your image
result = classify_custom_image("path/to/your/image.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Custom Categories

```python
# Define your own categories
custom_classes = [
    "a photo of a golden retriever",
    "a photo of a sports car",
    "a photo of a sunset landscape",
    "a photo of modern architecture"
]

# Use with the classification system
results = classify_with_custom_classes(image, custom_classes)
```

## 📊 Performance

On CIFAR-100 dataset:
- **Top-1 Accuracy**: ~62-70%
- **Top-5 Accuracy**: ~85-90%
- **Processing Speed**: ~10-50 images/second (depending on hardware)

## 🛠️ Supported CLIP Models

- `ViT-B/32` (Default) - Good balance of speed and accuracy
- `ViT-B/16` - Higher accuracy, slower processing
- `ViT-L/14` - Best accuracy, slowest processing
- `RN50` - ResNet-based, faster on some hardware

## 📈 Results and Visualization

The system generates:
- Accuracy metrics (Top-1, Top-5)
- Confusion matrices
- Per-class performance reports
- Classification confidence distributions
- Results saved in JSON format

## 🔍 Advanced Features

### Batch Processing
```python
# Process multiple images
results = batch_classify(image_paths, class_descriptions)
```

### Custom Prompts
```python
# Use detailed prompts for better accuracy
prompts = [
    "a professional photo of a {class_name}",
    "a {class_name} in natural lighting",
    "a clear image of a {class_name}"
]
```

### Performance Optimization
- GPU acceleration support
- Batch processing for efficiency
- Configurable image preprocessing
- Memory optimization for large datasets

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenAI CLIP
- PIL/Pillow
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the amazing pre-trained model
- CIFAR-100 dataset for evaluation benchmarks
- PyTorch team for the deep learning framework

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/Zero-Shot-Image-Classification-CLIP](https://github.com/yourusername/Zero-Shot-Image-Classification-CLIP)

---

⭐ **Star this repository if you found it helpful!**
