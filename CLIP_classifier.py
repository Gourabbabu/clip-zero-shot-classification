import streamlit as st
from PIL import Image
import torch
import clip
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from torchvision import transforms
import torch.nn.functional as F

# Page configuration
st.set_page_config(
    page_title="🎯 CIFAR-100 CLIP Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .hierarchical-card {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .accuracy-badge {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .dataset-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .added-classes {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for user-added classes
if 'user_added_classes' not in st.session_state:
    st.session_state.user_added_classes = {}

# CIFAR-100 class names with better descriptions + Extended classes for better coverage
CIFAR100_CLASSES = {
    'apple': 'a photo of an apple fruit',
    'aquarium_fish': 'a photo of an aquarium fish',
    'baby': 'a photo of a human baby',
    'bear': 'a photo of a bear',
    'beaver': 'a photo of a beaver',
    'bed': 'a photo of a bed',
    'bee': 'a photo of a bee',
    'beetle': 'a photo of a beetle',
    'bicycle': 'a photo of a bicycle',
    'bottle': 'a photo of a bottle',
    'bowl': 'a photo of a bowl',
    'boy': 'a photo of a boy',
    'bridge': 'a photo of a bridge',
    'bus': 'a photo of a bus',
    'butterfly': 'a photo of a butterfly',
    'camel': 'a photo of a camel',
    'can': 'a photo of a can',
    'castle': 'a photo of a castle',
    'caterpillar': 'a photo of a caterpillar',
    'cattle': 'a photo of cattle',
    'chair': 'a photo of a chair',
    'chimpanzee': 'a photo of a chimpanzee',
    'clock': 'a photo of a clock',
    'cloud': 'a photo of clouds',
    'cockroach': 'a photo of a cockroach',
    'couch': 'a photo of a couch',
    'crab': 'a photo of a crab',
    'crocodile': 'a photo of a crocodile',
    'cup': 'a photo of a cup',
    'dinosaur': 'a photo of a dinosaur',
    'dolphin': 'a photo of a dolphin',
    'elephant': 'a photo of an elephant',
    'flatfish': 'a photo of a flatfish',
    'forest': 'a photo of a forest',
    'fox': 'a photo of a fox',
    'girl': 'a photo of a girl',
    'hamster': 'a photo of a hamster',
    'house': 'a photo of a house',
    'kangaroo': 'a photo of a kangaroo',
    'keyboard': 'a photo of a computer keyboard',
    'lamp': 'a photo of a lamp',
    'lawn_mower': 'a photo of a lawn mower',
    'leopard': 'a photo of a leopard',
    'lion': 'a photo of a lion',
    'lizard': 'a photo of a lizard',
    'lobster': 'a photo of a lobster',
    'man': 'a photo of a man',
    'maple_tree': 'a photo of a maple tree',
    'motorcycle': 'a photo of a motorcycle',
    'mountain': 'a photo of a mountain',
    'mouse': 'a photo of a mouse',
    'mushroom': 'a photo of a mushroom',
    'oak_tree': 'a photo of an oak tree',
    'orange': 'a photo of an orange fruit',
    'orchid': 'a photo of an orchid flower',
    'otter': 'a photo of an otter',
    'palm_tree': 'a photo of a palm tree',
    'pear': 'a photo of a pear fruit',
    'pickup_truck': 'a photo of a pickup truck',
    'pine_tree': 'a photo of a pine tree',
    'plain': 'a photo of a plain landscape',
    'plate': 'a photo of a plate',
    'poppy': 'a photo of a poppy flower',
    'porcupine': 'a photo of a porcupine',
    'possum': 'a photo of a possum',
    'rabbit': 'a photo of a rabbit',
    'raccoon': 'a photo of a raccoon',
    'ray': 'a photo of a ray fish',
    'road': 'a photo of a road',
    'rocket': 'a photo of a rocket',
    'rose': 'a photo of a rose flower',
    'sea': 'a photo of the sea',
    'seal': 'a photo of a seal',
    'shark': 'a photo of a shark',
    'shrew': 'a photo of a shrew',
    'skunk': 'a photo of a skunk',
    'skyscraper': 'a photo of a skyscraper',
    'snail': 'a photo of a snail',
    'snake': 'a photo of a snake',
    'spider': 'a photo of a spider',
    'squirrel': 'a photo of a squirrel',
    'streetcar': 'a photo of a streetcar',
    'sunflower': 'a photo of a sunflower',
    'sweet_pepper': 'a photo of a sweet pepper',
    'table': 'a photo of a table',
    'tank': 'a photo of a tank',
    'telephone': 'a photo of a telephone',
    'television': 'a photo of a television',
    'tiger': 'a photo of a tiger',
    'tractor': 'a photo of a tractor',
    'train': 'a photo of a train',
    'trout': 'a photo of a trout fish',
    'tulip': 'a photo of a tulip flower',
    'turtle': 'a photo of a turtle',
    'wardrobe': 'a photo of a wardrobe',
    'whale': 'a photo of a whale',
    'willow_tree': 'a photo of a willow tree',
    'wolf': 'a photo of a wolf',
    'woman': 'a photo of a woman',
    'worm': 'a photo of a worm',
    
    # Extended classes for better animal coverage
    'koala': 'a photo of a koala bear',
    'panda': 'a photo of a giant panda',
    'sloth': 'a photo of a sloth',
    'penguin': 'a photo of a penguin',
    'flamingo': 'a photo of a flamingo',
    'giraffe': 'a photo of a giraffe',
    'zebra': 'a photo of a zebra',
    'hippo': 'a photo of a hippopotamus',
    'rhino': 'a photo of a rhinoceros',
    'ostrich': 'a photo of an ostrich',
    'peacock': 'a photo of a peacock',
    'swan': 'a photo of a swan',
    'owl': 'a photo of an owl',
    'eagle': 'a photo of an eagle',
    'parrot': 'a photo of a parrot'
}

# Add user-added classes to the main dictionary
for class_name, description in st.session_state.user_added_classes.items():
    CIFAR100_CLASSES[class_name] = description

# Hierarchical category definitions
HIERARCHICAL_CATEGORIES = {
    'Animal': [
        'bear', 'beaver', 'camel', 'cattle', 'chimpanzee', 'elephant', 'fox', 'hamster', 
        'kangaroo', 'leopard', 'lion', 'mouse', 'otter', 'porcupine', 'possum', 'rabbit', 
        'raccoon', 'seal', 'shrew', 'skunk', 'squirrel', 'tiger', 'wolf', 'koala', 'panda', 
        'sloth', 'giraffe', 'zebra', 'hippo', 'rhino'
    ],
    'Bird': [
        'penguin', 'flamingo', 'ostrich', 'peacock', 'swan', 'owl', 'eagle', 'parrot'
    ],
    'Fish': [
        'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'
    ],
    'Insect': [
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'
    ],
    'Invertebrate': [
        'crab', 'lobster', 'snail', 'spider', 'worm'
    ],
    'Reptile': [
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'
    ],
    'Marine Animal': [
        'dolphin', 'whale', 'beaver', 'otter', 'seal'
    ],
    'Human': [
        'baby', 'boy', 'girl', 'man', 'woman'
    ],
    'Plant': [
        'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'orchid', 
        'poppy', 'rose', 'sunflower', 'tulip', 'mushroom'
    ],
    'Fruit': [
        'apple', 'orange', 'pear'
    ],
    'Vegetable': [
        'sweet_pepper'
    ],
    'Vehicle': [
        'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 
        'rocket', 'streetcar', 'tank', 'tractor'
    ],
    'Furniture': [
        'bed', 'chair', 'couch', 'table', 'wardrobe'
    ],
    'Container': [
        'bottle', 'bowl', 'can', 'cup', 'plate'
    ],
    'Electronic Device': [
        'clock', 'keyboard', 'lamp', 'telephone', 'television'
    ],
    'Building/Structure': [
        'bridge', 'castle', 'house', 'skyscraper'
    ],
    'Natural Landscape': [
        'cloud', 'forest', 'mountain', 'plain', 'sea', 'road'
    ]
}

# Add user-added classes to hierarchical categories
for class_name, category in st.session_state.user_added_classes.items():
    if class_name not in HIERARCHICAL_CATEGORIES.get(category, []):
        if category not in HIERARCHICAL_CATEGORIES:
            HIERARCHICAL_CATEGORIES[category] = []
        HIERARCHICAL_CATEGORIES[category].append(class_name)

# Broad category text embeddings
BROAD_CATEGORIES = {
    'a photo of an animal',
    'a photo of a bird', 
    'a photo of a fish',
    'a photo of an insect',
    'a photo of an invertebrate',
    'a photo of a reptile',
    'a photo of a marine animal',
    'a photo of a human person',
    'a photo of a plant',
    'a photo of a fruit',
    'a photo of a vegetable',
    'a photo of a vehicle',
    'a photo of furniture',
    'a photo of a container',
    'a photo of an electronic device',
    'a photo of a building or structure',
    'a photo of a natural landscape'
}

# Load CLIP model once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_data
def load_cifar100_embeddings():
    """Pre-compute text embeddings for CIFAR-100 classes"""
    model, _, device = load_model()
    
    class_names = list(CIFAR100_CLASSES.keys())
    descriptions = list(CIFAR100_CLASSES.values())
    
    # Tokenize all descriptions
    text_inputs = clip.tokenize(descriptions).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features, class_names, descriptions

@st.cache_data
def load_hierarchical_embeddings():
    """Pre-compute text embeddings for broad categories"""
    model, _, device = load_model()
    
    broad_categories_list = list(BROAD_CATEGORIES)
    text_inputs = clip.tokenize(broad_categories_list).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features, broad_categories_list

def get_hierarchical_classification(class_name, confidence):
    """Get hierarchical classification based on the predicted class and confidence"""
    hierarchical_results = []
    
    for category, classes in HIERARCHICAL_CATEGORIES.items():
        if class_name in classes:
            hierarchical_results.append({
                'category': category,
                'confidence': confidence,
                'emoji': get_category_emoji(category)
            })
    
    return hierarchical_results

def get_category_emoji(category):
    """Get emoji for each category"""
    emoji_map = {
        'Animal': '🐾',
        'Bird': '🦅', 
        'Fish': '🐟',
        'Insect': '🐛',
        'Invertebrate': '🦗',
        'Reptile': '🦎',
        'Marine Animal': '🌊',
        'Human': '👤',
        'Plant': '🌱',
        'Fruit': '🍎',
        'Vegetable': '🥬',
        'Vehicle': '🚗',
        'Furniture': '🪑',
        'Container': '🫙',
        'Electronic Device': '📱',
        'Building/Structure': '🏢',
        'Natural Landscape': '🌄'
    }
    return emoji_map.get(category, '📦')

def perform_broad_classification(image_features):
    """Perform broad category classification when specific confidence is low"""
    broad_text_features, broad_categories = load_hierarchical_embeddings()
    
    with torch.no_grad():
        similarities = (image_features @ broad_text_features.T).softmax(dim=-1)
        probs = similarities.cpu().numpy()[0]
    
    return broad_categories, probs

def create_enhanced_confidence_chart(labels, probabilities, top_n=15):
    """Create an enhanced confidence chart with CIFAR-100 styling"""
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_labels = [labels[i].replace('_', ' ').title() for i in top_indices]
    top_probs = [probabilities[i] * 100 for i in top_indices]
    
    # Create color gradient based on confidence
    colors = []
    for prob in top_probs:
        if prob > 50:
            colors.append('#1f77b4')  # Strong blue
        elif prob > 20:
            colors.append('#ff7f0e')  # Orange
        elif prob > 10:
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_labels[::-1],
            x=top_probs[::-1],
            orientation='h',
            marker=dict(
                color=colors[::-1],
                line=dict(color='rgba(0, 0, 0, 0.8)', width=1)
            ),
            text=[f'{prob:.1f}%' for prob in top_probs[::-1]],
            textposition='inside',
            textfont=dict(color='white', size=11, family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='🎯 CIFAR-100 Classification Confidence',
            x=0.5,
            font=dict(size=18, family='Arial Black')
        ),
        xaxis_title='Confidence (%)',
        yaxis_title='CIFAR-100 Classes',
        template='plotly_white',
        height=max(500, len(top_labels) * 35),
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    return fig

def get_superclass_info():
    """Get CIFAR-100 superclass information"""
    superclasses = {
        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'extended_animals': ['koala', 'panda', 'sloth', 'giraffe', 'zebra', 'hippo', 'rhino'],
        'birds': ['penguin', 'flamingo', 'ostrich', 'peacock', 'swan', 'owl', 'eagle', 'parrot'],
        'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    }
    return superclasses

# Initialize model and embeddings
model, preprocess, device = load_model()
text_features, class_names, descriptions = load_cifar100_embeddings()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem; margin: 0;">🎯 Enhanced CIFAR-100 CLIP Classifier</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        With Hierarchical Classification • Identifies broad categories when specific classification is uncertain
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Enhanced Features")
    
    # Dataset info
    st.markdown("""
    <div class="dataset-info">
        <h4>🧠 Smart Classification</h4>
        <p><strong>Specific:</strong> 115 detailed classes</p>
        <p><strong>Hierarchical:</strong> 17 broad categories</p>
        <p><strong>Fallback:</strong> Animal, Plant, Object, etc.</p>
        <p><strong>Confidence threshold:</strong> Dynamic</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    st.info(f"🖥️ **Device:** {device.upper()}\n🧠 **Model:** CLIP ViT-B/32\n✨ **Enhanced:** Hierarchical Classification")
    
    # Display options
    st.markdown("### 📊 Display Options")
    top_n = st.slider("Top predictions to show", 5, 25, 15)
    confidence_threshold = st.slider("Confidence threshold for hierarchical classification", 10, 80, 30)
    show_superclass = st.checkbox("🏷️ Show superclass info", value=True)
    show_hierarchical = st.checkbox("🌳 Show hierarchical classification", value=True)
    show_chart = st.checkbox("📈 Show confidence chart", value=True)
    show_all_predictions = st.checkbox("📋 Show all predictions", value=False)
    
    # Add New Class Section
    st.markdown("---")
    st.markdown("### 🆕 Add New Animal/Bird")
    
    new_class_name = st.text_input("Class Name (e.g., 'koala')")
    new_class_desc = st.text_input("Description (e.g., 'a photo of a koala bear')")
    new_class_category = st.selectbox("Category", sorted(list(HIERARCHICAL_CATEGORIES.keys())))
    
    if st.button("Add New Class"):
        if new_class_name and new_class_desc and new_class_category:
            # Add to class dictionary
            st.session_state.user_added_classes[new_class_name] = new_class_desc
            
            # Add to hierarchical categories
            if new_class_name not in HIERARCHICAL_CATEGORIES.get(new_class_category, []):
                if new_class_category not in HIERARCHICAL_CATEGORIES:
                    HIERARCHICAL_CATEGORIES[new_class_category] = []
                HIERARCHICAL_CATEGORIES[new_class_category].append(new_class_name)
            
            # Clear embeddings cache to include new class
            st.cache_data.clear()
            st.success(f"✅ Added '{new_class_name}' to '{new_class_category}' category!")
            st.rerun()
        else:
            st.warning("Please fill all fields")
    
    # Show added classes
    if st.session_state.user_added_classes:
        st.markdown("### 🆕 User-Added Classes")
        st.markdown('<div class="added-classes">', unsafe_allow_html=True)
        for class_name, category in st.session_state.user_added_classes.items():
            st.markdown(f"🐾 **{class_name}** → {category}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Reset Added Classes"):
            st.session_state.user_added_classes = {}
            st.cache_data.clear()
            st.success("✅ Reset done! Refreshing...")
            st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 style="color: white; text-align: center;">📷 Upload Your Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="prediction-card">
                <h4>📋 Image Information</h4>
                <p><strong>Filename:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {image.size[0]} × {image.size[1]} pixels</p>
                <p><strong>Format:</strong> {image.format}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

with col2:
    if uploaded_file:
        try:
            with st.spinner("🤖 Analyzing with Enhanced CIFAR-100 CLIP..."):
                # Preprocess image
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = F.normalize(image_features, dim=-1)
                    
                    # Compute similarities with pre-computed text embeddings
                    similarities = (image_features @ text_features.T).softmax(dim=-1)
                    probs = similarities.cpu().numpy()[0]
            
            # Top prediction highlight
            top_idx = np.argmax(probs)
            confidence = probs[top_idx] * 100
            predicted_class = class_names[top_idx]
            
            # Get superclass info
            superclasses = get_superclass_info()
            predicted_superclass = None
            for superclass, classes in superclasses.items():
                if predicted_class in classes:
                    predicted_superclass = superclass.replace('_', ' ').title()
                    break
            
            # Show specific classification
            st.markdown(f"""
            <div class="metric-card">
                <h2>🏆 Best Specific Match</h2>
                <h3>{predicted_class.replace('_', ' ').title()}</h3>
                <h1>{confidence:.1f}%</h1>
                {f'<p>📂 Superclass: {predicted_superclass}</p>' if predicted_superclass and show_superclass else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Hierarchical classification when confidence is low or always show if enabled
            if show_hierarchical and (confidence < confidence_threshold or st.sidebar.checkbox("Always show hierarchical", value=False)):
                hierarchical_results = get_hierarchical_classification(predicted_class, confidence)
                
                if hierarchical_results:
                    for result in hierarchical_results:
                        st.markdown(f"""
                        <div class="hierarchical-card">
                            <h2>{result['emoji']} Hierarchical Classification</h2>
                            <h3>This appears to be: {result['category']}</h3>
                            <p>Based on specific classification confidence: {result['confidence']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Perform broad classification for additional context
                if confidence < confidence_threshold:
                    with st.spinner("🔍 Performing broad category analysis..."):
                        broad_categories, broad_probs = perform_broad_classification(image_features)
                        
                        top_broad_idx = np.argmax(broad_probs)
                        broad_confidence = broad_probs[top_broad_idx] * 100
                        broad_category = broad_categories[top_broad_idx]
                        
                        st.markdown(f"""
                        <div class="hierarchical-card">
                            <h2>🎯 Broad Category Analysis</h2>
                            <h3>{broad_category.replace('a photo of ', '').title()}</h3>
                            <h1>{broad_confidence:.1f}%</h1>
                            <p>When specific classification is uncertain, this provides general category</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Accuracy indicator
            if confidence > 70:
                accuracy_color = "#4CAF50"
                accuracy_text = "High Confidence"
            elif confidence > 40:
                accuracy_color = "#FF9800" 
                accuracy_text = "Medium Confidence"
            else:
                accuracy_color = "#F44336"
                accuracy_text = "Low Confidence - Check Hierarchical"
                
            st.markdown(f"""
            <div style="background: {accuracy_color}; padding: 0.5rem; border-radius: 10px; text-align: center; color: white; font-weight: bold; margin: 1rem 0;">
                {accuracy_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Top N predictions
            top_indices = np.argsort(probs)[::-1][:top_n]
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Top {top_n} CIFAR-100 Predictions")
            
            for i, idx in enumerate(top_indices):
                pred_confidence = probs[idx] * 100
                class_name = class_names[idx]
                label = class_name.replace('_', ' ').title()
                
                # Get superclass for this prediction
                pred_superclass = None
                if show_superclass:
                    for superclass, classes in superclasses.items():
                        if class_name in classes:
                            pred_superclass = superclass.replace('_', ' ').title()
                            break
                
                # Get hierarchical info for this prediction
                hierarchical_info = get_hierarchical_classification(class_name, pred_confidence)
                hierarchical_text = ""
                if hierarchical_info and show_hierarchical:
                    categories = [f"{r['emoji']}{r['category']}" for r in hierarchical_info]
                    hierarchical_text = f" • {', '.join(categories)}"
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span><strong>{i+1}. {label}</strong> {f'({pred_superclass})' if pred_superclass and show_superclass else ''}{hierarchical_text}</span>
                        <span><strong>{pred_confidence:.2f}%</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(pred_confidence / 100)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced confidence chart
            if show_chart:
                fig = create_enhanced_confidence_chart(class_names, probs, top_n)
                st.plotly_chart(fig, use_container_width=True)
            
            # All predictions with superclass grouping
            if show_all_predictions:
                st.markdown("### 📊 All CIFAR-100 Predictions")
                
                all_predictions = []
                for i, prob in enumerate(probs):
                    class_name = class_names[i]
                    superclass = None
                    for sc, classes in superclasses.items():
                        if class_name in classes:
                            superclass = sc.replace('_', ' ').title()
                            break
                    
                    # Get hierarchical categories
                    hierarchical_cats = get_hierarchical_classification(class_name, prob * 100)
                    hierarchical_str = ', '.join([cat['category'] for cat in hierarchical_cats]) if hierarchical_cats else 'None'
                    
                    all_predictions.append({
                        'Class': class_name.replace('_', ' ').title(),
                        'Superclass': superclass or 'Unknown',
                        'Hierarchical Categories': hierarchical_str,
                        'Confidence (%)': f"{prob * 100:.3f}"
                    })
                
                df = pd.DataFrame(all_predictions)
                df = df.sort_values('Confidence (%)', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced CIFAR-100 Results",
                    data=csv,
                    file_name=f"enhanced_cifar100_classification_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("💡 Make sure the image is clear and try again.")
    
    else:
        st.markdown("""
        <div class="prediction-card" style="text-align: center; padding: 3rem;">
            <h3>🚀 Ready for Enhanced Classification!</h3>
            <p>Upload an image for intelligent hierarchical classification.</p>
            <p><strong>Features:</strong></p>
            <ul style="text-align: left; max-width: 300px; margin: 0 auto;">
                <li>🎯 Specific CIFAR-100 classification</li>
                <li>🌳 Hierarchical category fallback</li>
                <li>🐾 Animal/Plant/Object identification</li>
                <li>🧠 Smart confidence thresholds</li>
                <li>🆕 Add new animals/birds on-the-fly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer with enhanced info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p>🎯 <strong>Enhanced CIFAR-100 Classifier with Hierarchical Intelligence</strong></p>
    <p>🧠 When specific classification is uncertain, it identifies broad categories like "Animal", "Plant", "Vehicle", etc.</p>
    <p>📊 Smart fallback system ensures meaningful results even for unknown objects</p>
    <p>🆕 Add new animals/birds just by entering their name and description!</p>
</div>
""", unsafe_allow_html=True)
