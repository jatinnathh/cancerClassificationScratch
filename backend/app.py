"""
Flask API for Cancer Classification
Supports: Brain Tumor, Lung Cancer, Skin Cancer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
from skimage import exposure
from skimage.morphology import binary_closing, binary_opening, disk
import io
import pickle
import os
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).resolve().parent  # Get the directory where app.py is located
UPLOAD_FOLDER = BASE_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure device - prioritize CUDA GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"Device Configuration:")
print(f"{'='*60}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"GPU Available: ✓ ENABLED")
else:
    print(f"GPU Available: ✗ Running on CPU (slower)")
print(f"{'='*60}\n")

# ============================================
# MODEL ARCHITECTURES
# ============================================

class ImprovedBrainTumorCNN(nn.Module):
    """Improved Brain Tumor CNN with better regularization"""
    def __init__(self, num_classes=4):
        super(ImprovedBrainTumorCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # IMPROVED classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(512, 256),  # Added intermediate layer
            nn.BatchNorm1d(256),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Additional dropout
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# Keep old architecture for backward compatibility
class BrainTumorCNN(nn.Module):
    """Legacy Brain Tumor CNN - kept for backward compatibility"""
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class LungCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# ============================================
# BRAIN TUMOR PREPROCESSING FUNCTIONS
# ============================================

def adjust_gamma(image, gamma=1.0):
    """Adjust gamma of the image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def correct_bias_field(image):
    """Approximate bias field correction using local contrast enhancement."""
    img_float = image.astype(np.float32) / 255.0
    bias_field = cv2.GaussianBlur(img_float, (99, 99), 0)
    bias_field = np.maximum(bias_field, 0.01)
    corrected = img_float / bias_field
    corrected = exposure.rescale_intensity(corrected, out_range=(0, 1))
    corrected = (corrected * 255).astype(np.uint8)
    return corrected

def preprocess_brain_image(image):
    """Preprocess the image with improved bias correction and contrast enhancement."""
    image_float = image.astype(float)
    image_norm = (image_float - image_float.min()) / (image_float.max() - image_float.min())
    image_norm = (image_norm * 255).astype(np.uint8)
    image_bias_corrected = correct_bias_field(image_norm)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image_bias_corrected)
    image_gamma = adjust_gamma(image_clahe, gamma=1.2)
    image_denoised = cv2.fastNlMeansDenoising(image_gamma)
    return image_denoised

def create_brain_mask(image):
    """Create a binary mask for brain region."""
    image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    selem_main = disk(5)
    mask = binary_closing(binary_opening(binary > 0, selem_main), selem_main)
    mask = (mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth)
    return mask

def apply_mask(image, mask):
    """Apply the mask to the original image."""
    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edge_mask = cv2.dilate(mask, kernel_edge, iterations=1)
    inner_mask = cv2.erode(mask, kernel_edge, iterations=1)
    edge_zone = cv2.subtract(edge_mask, inner_mask)
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    edge_pixels = cv2.bitwise_and(image, image, mask=edge_zone)
    edge_pixels = cv2.GaussianBlur(edge_pixels, (3,3), 0)
    
    final_image = cv2.add(
        cv2.bitwise_and(masked_image, masked_image, mask=cv2.bitwise_not(edge_zone)),
        edge_pixels
    )
    return final_image


# ============================================
# LOAD MODELS
# ============================================

def load_models():
    """Load all three models"""
    models = {}
    
    # Load Brain Tumor Model (Try improved v2 first, fallback to v1)
    try:
        # Try to load the improved v2 model first
        brain_model_path = BASE_DIR / 'src' / 'brain' / 'brain_tumor_classifier_v2_improved.pth'
        
        # if brain_model_path.exists():
        #     print("Loading IMPROVED Brain Tumor Model v2...")
        #     brain_checkpoint = torch.load(str(brain_model_path), map_location=device)
        #     brain_model = ImprovedBrainTumorCNN(num_classes=4)
        #     brain_model.load_state_dict(brain_checkpoint['model_state_dict'])
        #     brain_model = brain_model.to(device)
        #     brain_model.eval()
        #     models['brain'] = {
        #         'model': brain_model,
        #         'classes': brain_checkpoint['model_config']['class_names'],
        #         'input_size': brain_checkpoint['preprocessing']['input_size'],
        #         'mean': brain_checkpoint['preprocessing']['mean'],
        #         'std': brain_checkpoint['preprocessing']['std'],
        #         'version': 'v2_improved',
        #         'test_accuracy': brain_checkpoint['performance']['best_test_accuracy']
        #     }
        #     print(f"✓ Brain tumor model v2 loaded (Test Acc: {brain_checkpoint['performance']['best_test_accuracy']:.2%})")
        # else:
            # Fallback to old v1 model
        
        print("v2 model not found, loading legacy v1 model...")
        brain_model_path = BASE_DIR / 'src' / 'brain' / 'brain_tumor_classifier_v1.pth'
        brain_checkpoint = torch.load(str(brain_model_path), map_location=device)
        brain_model = BrainTumorCNN(num_classes=4)
        brain_model.load_state_dict(brain_checkpoint['model_state_dict'])
        brain_model = brain_model.to(device)
        brain_model.eval()
        models['brain'] = {
            'model': brain_model,
            'classes': brain_checkpoint['model_config']['class_names'],
            'input_size': brain_checkpoint['preprocessing']['input_size'],
            'mean': brain_checkpoint['preprocessing']['mean'],
            'std': brain_checkpoint['preprocessing']['std'],
            'version': 'v1_legacy',
            'test_accuracy': brain_checkpoint['performance'].get('best_val_accuracy', 0)
        }
        print(f"✓ Brain tumor model v1 loaded (Val Acc: {brain_checkpoint['performance']['best_val_accuracy']:.2%})")
        print("⚠️  Note: Using legacy model. Train improved model for better accuracy!")
    except Exception as e:
            print(f"✗ Error loading brain model: {e}")
    
    # Load Lung Cancer Model (using checkpoint approach from testl.ipynb)
    try:
        lung_checkpoint_path = BASE_DIR / 'src' / 'lungs' / 'lung_cnn_checkpoint.pth'
        lung_classes_path = BASE_DIR / 'src' / 'lungs' / 'lung_class_names.pkl'
        
        # Load checkpoint
        lung_checkpoint = torch.load(str(lung_checkpoint_path), map_location=device)
        
        # Initialize model with LungCNN architecture
        lung_model = LungCNN(num_classes=lung_checkpoint['num_classes'])
        lung_model.load_state_dict(lung_checkpoint['model_state_dict'])
        lung_model = lung_model.to(device)
        lung_model.eval()
        
        # Load class names from checkpoint or pickle file
        if 'class_names' in lung_checkpoint:
            lung_classes = lung_checkpoint['class_names']
        else:
            with open(str(lung_classes_path), 'rb') as f:
                lung_classes = pickle.load(f)
        
        models['lung'] = {
            'model': lung_model,
            'classes': lung_classes,
            'input_size': lung_checkpoint.get('input_size', (224, 224)),
            'mean': lung_checkpoint.get('normalize_mean', [0.485, 0.456, 0.406]),
            'std': lung_checkpoint.get('normalize_std', [0.229, 0.224, 0.225]),
            'test_accuracy': lung_checkpoint.get('test_acc', 'N/A'),
            'best_val_acc': lung_checkpoint.get('best_val_acc', 'N/A')
        }
        print(f"✓ Lung cancer model loaded (LungCNN)")
        if 'best_val_acc' in lung_checkpoint:
            print(f"  Best Val Acc: {lung_checkpoint['best_val_acc']:.4f}")
        if 'test_acc' in lung_checkpoint:
            print(f"  Test Acc: {lung_checkpoint['test_acc']:.4f}")
    except Exception as e:
        print(f"✗ Error loading lung model: {e}")
        print(f"  Make sure 'lung_cnn_checkpoint.pth' exists in {BASE_DIR / 'src' / 'lungs' / ''}")
    
    # Load Skin Cancer Model
    try:
        skin_model_path = BASE_DIR / 'src' / 'skin' / 'skin_cnn_full_model.pth'
        skin_classes_path = BASE_DIR / 'src' / 'skin' / 'class_names.pkl'
        
        skin_model = torch.load(str(skin_model_path), map_location=device)
        skin_model.eval()
        
        with open(str(skin_classes_path), 'rb') as f:
            skin_classes = pickle.load(f)
        
        models['skin'] = {
            'model': skin_model,
            'classes': skin_classes,
            'input_size': 128,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        print("✓ Skin cancer model loaded")
    except Exception as e:
        print(f"✗ Error loading skin model: {e}")
    
    return models

# Initialize models
MODELS = load_models()


# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_brain_tumor(image_file):
    """Predict brain tumor type"""
    if 'brain' not in MODELS:
        return {'error': 'Brain tumor model not loaded'}
    
    model_info = MODELS['brain']
    model = model_info['model']
    
    # Read image
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    
    # Convert to grayscale for preprocessing
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply brain mask preprocessing
    preprocessed = preprocess_brain_image(img_gray)
    mask = create_brain_mask(preprocessed)
    masked = apply_mask(preprocessed, mask)
    img_processed = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL
    img_pil = Image.fromarray(img_processed)
    
    # Handle input_size - could be int or tuple
    input_size = model_info['input_size']
    if isinstance(input_size, (list, tuple)):
        resize_size = input_size[0] if len(input_size) > 0 else 224
    else:
        resize_size = input_size
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_info['mean'], std=model_info['std'])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = model_info['classes'][predicted.item()]
    confidence_score = confidence.item() * 100
    
    all_probs = {model_info['classes'][i]: probabilities[0][i].item() * 100 
                 for i in range(len(model_info['classes']))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'all_probabilities': all_probs,
        'cancer_type': 'Brain Tumor'
    }


def predict_lung_cancer(image_file):
    """Predict lung cancer type"""
    if 'lung' not in MODELS:
        return {'error': 'Lung cancer model not loaded'}
    
    model_info = MODELS['lung']
    model = model_info['model']
    
    # Load and transform image
    image = Image.open(image_file).convert('RGB')
    
    # Handle input_size - could be int or tuple
    input_size = model_info['input_size']
    if isinstance(input_size, (list, tuple)):
        resize_size = input_size[0] if len(input_size) > 0 else 224
    else:
        resize_size = input_size
    
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_info['mean'], std=model_info['std'])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = model_info['classes'][predicted.item()]
    confidence_score = confidence.item() * 100
    
    all_probs = {model_info['classes'][i]: probabilities[0][i].item() * 100 
                 for i in range(len(model_info['classes']))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'all_probabilities': all_probs,
        'cancer_type': 'Lung Cancer'
    }


def predict_skin_cancer(image_file):
    """Predict skin cancer type"""
    if 'skin' not in MODELS:
        return {'error': 'Skin cancer model not loaded'}
    
    model_info = MODELS['skin']
    model = model_info['model']
    
    # Load and transform image
    image = Image.open(image_file).convert('RGB')
    
    # Handle input_size - could be int or tuple
    input_size = model_info['input_size']
    if isinstance(input_size, (list, tuple)):
        resize_size = input_size[0] if len(input_size) > 0 else 128
    else:
        resize_size = input_size
    
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_info['mean'], std=model_info['std'])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = model_info['classes'][predicted.item()]
    confidence_score = confidence.item() * 100
    
    all_probs = {model_info['classes'][i]: probabilities[0][i].item() * 100 
                 for i in range(len(model_info['classes']))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'all_probabilities': all_probs,
        'cancer_type': 'Skin Cancer'
    }


# ============================================
# API ROUTES
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(MODELS.keys()),
        'device': str(device)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint
    Expects: file (image) and cancer_type (brain/lung/skin)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Get cancer type
        cancer_type = request.form.get('cancer_type', '').lower()
        
        if cancer_type not in ['brain', 'lung', 'skin']:
            return jsonify({'error': 'Invalid cancer_type. Must be: brain, lung, or skin'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = str(UPLOAD_FOLDER / filename)
        file.save(filepath)
        
        # Make prediction based on cancer type
        if cancer_type == 'brain':
            result = predict_brain_tumor(filepath)
        elif cancer_type == 'lung':
            result = predict_lung_cancer(filepath)
        elif cancer_type == 'skin':
            result = predict_skin_cancer(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models"""
    models_info = {}
    for key, value in MODELS.items():
        models_info[key] = {
            'classes': value['classes'],
            'num_classes': len(value['classes']),
            'input_size': value['input_size'],
            'version': value.get('version', 'unknown'),
            'test_accuracy': value.get('test_accuracy', 'N/A')
        }
    return jsonify(models_info)


@app.route('/api/models/brain/info', methods=['GET'])
def get_brain_model_info():
    """Get detailed information about the brain tumor model"""
    if 'brain' not in MODELS:
        return jsonify({'error': 'Brain model not loaded'}), 404
    
    brain_info = MODELS['brain']
    return jsonify({
        'version': brain_info.get('version', 'unknown'),
        'classes': brain_info['classes'],
        'num_classes': len(brain_info['classes']),
        'input_size': brain_info['input_size'],
        'test_accuracy': brain_info.get('test_accuracy', 'N/A'),
        'normalization': {
            'mean': brain_info['mean'],
            'std': brain_info['std']
        },
        'architecture': 'ImprovedBrainTumorCNN' if brain_info.get('version') == 'v2_improved' else 'BrainTumorCNN (Legacy)',
        'preprocessing': ['Bias correction', 'CLAHE', 'Gamma adjustment', 'Denoising', 'Brain masking']
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Cancer Classification API Server")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models loaded: {list(MODELS.keys())}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
