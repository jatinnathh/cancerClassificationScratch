# Brain Tumor Model - Diagnosis and Fix Plan

## 🚨 Critical Issues Discovered

### Test Performance
- **Validation Accuracy**: 96.35% ✓
- **Actual Test Accuracy**: 53.75% ✗
- **Gap**: 42.6% (SEVERE OVERFITTING)

### Per-Class Performance
| Class | Accuracy | Issue |
|-------|----------|-------|
| glioma | 90% | ✓ Good |
| meningioma | 40% | ✗ Often confused with glioma |
| notumor | 25% | ✗ Predicted as glioma/meningioma |
| pituitary | 60% | ⚠️ Confused with meningioma |

### Key Problems

1. **Severe Overfitting**
   - Model memorized validation set
   - Fails to generalize to new test data
   - 42.6% accuracy gap indicates overfitting

2. **Class Imbalance**
   - "notumor" only 25% accuracy
   - Model biased toward tumor classes
   - Need to check class distribution

3. **Aggressive Augmentation**
   - May be distorting important medical features
   - Perspective transforms inappropriate for MRI scans
   - Excessive zoom/shear

4. **Data Split Issues**
   - Using `random_split` on ImageFolder might not ensure proper separation
   - Validation set might be too similar to training

## 🔧 Fix Strategy

### Priority 1: Reduce Overfitting

#### A. Better Data Splitting
```python
# Instead of random_split, use stratified split with fixed seed
from sklearn.model_selection import train_test_split

# Get all file paths and labels
all_files = []
all_labels = []
for class_idx, class_name in enumerate(full_dataset.classes):
    class_dir = Path(data_dir) / class_name
    class_files = list(class_dir.glob('*.jpg'))
    all_files.extend(class_files)
    all_labels.extend([class_idx] * len(class_files))

# Stratified split
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)
```

#### B. Reduce Augmentation Intensity
```python
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),  # Reduced from 45
    transforms.RandomResizedCrop((128, 128), scale=(0.85, 1.15)),  # Reduced from (0.7, 1.3)
    transforms.RandomHorizontalFlip(p=0.5),
    # REMOVED: RandomVerticalFlip (not realistic for axial MRI)
    # REMOVED: RandomPerspective (distorts anatomy)
    # REMOVED: RandomAffine with shear (too aggressive)
    transforms.ColorJitter(
        brightness=0.15,  # Reduced from 0.3
        contrast=0.15,    # Reduced from 0.3
        saturation=0,     # MRI is grayscale, remove saturation
        hue=0             # MRI is grayscale, remove hue
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    GaussianNoise(0., 0.005)  # Reduced from 0.01
])
```

#### C. Add Regularization
```python
# Increase dropout
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.6),  # Increased from 0.5
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),  # Add another dropout layer
    nn.Linear(256, num_classes)
)

# Increase weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)  # Increased from 0.01
```

#### D. Early Stopping
```python
# Reduce patience (already at 10, which is good)
# But monitor test accuracy, not just validation
patience = 7  # Reduce from 10
```

### Priority 2: Fix Class Imbalance

#### Check Class Distribution
```python
# Count samples per class
class_counts = {}
for class_name in full_dataset.classes:
    class_dir = Path(data_dir) / class_name
    count = len(list(class_dir.glob('*.jpg')))
    class_counts[class_name] = count
    print(f"{class_name}: {count} images")
```

#### Adjust Class Weights
```python
# Use class weights with stronger emphasis
class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

#### Add Class-Specific Data Augmentation
```python
# Apply more augmentation to minority classes
# Less augmentation to majority classes
```

### Priority 3: Improve Training Process

#### A. Use K-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(all_files, all_labels)):
    # Train model on this fold
    # Track performance
    pass
```

#### B. Add Test Set Evaluation
```python
# After each epoch, also evaluate on test set
# Monitor both val and test accuracy
# Stop if gap between val and test grows
```

#### C. Learning Rate Schedule
```python
# Use ReduceLROnPlateau instead of OneCycleLR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
```

### Priority 4: Model Architecture Changes

#### Consider Using Pre-trained Model
```python
import torchvision.models as models

# Use ResNet18 pre-trained on ImageNet
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)
```

#### Or Add Batch Normalization
```python
# Already have BatchNorm2d - good!
# Consider adding BatchNorm1d in classifier
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.BatchNorm1d(512),  # Add this
    nn.ReLU(inplace=True),
    nn.Dropout(0.6),
    nn.Linear(512, num_classes)
)
```

## 📊 Testing Recommendations

### 1. Calculate Actual Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions on test set
y_true = []
y_pred = []
# ... collect predictions ...

print(classification_report(y_true, y_pred, target_names=class_names))
```

### 2. Analyze Misclassifications
```python
# Look at images the model gets wrong
# Are they ambiguous even for humans?
# Do they have different preprocessing?
```

### 3. Check Data Quality
```python
# Verify train/val/test splits don't overlap
# Check if mask_cache preprocessing is consistent
# Ensure no data leakage
```

## 🎯 Immediate Action Plan

### Step 1: Quick Fix (1-2 hours)
1. ✅ Reduce augmentation intensity
2. ✅ Increase regularization (dropout, weight decay)
3. ✅ Add validation on actual test set
4. ✅ Retrain for 20 epochs

### Step 2: Better Data Split (2-3 hours)
1. Implement stratified split
2. Verify no data leakage
3. Ensure consistent preprocessing
4. Retrain and compare

### Step 3: Advanced Improvements (4-6 hours)
1. Try transfer learning (ResNet18/EfficientNet)
2. Implement K-fold cross-validation
3. Add class-specific augmentation
4. Ensemble multiple models

## 📝 Expected Improvements

| Change | Expected Test Accuracy |
|--------|----------------------|
| Current | 53.75% |
| After reducing augmentation | 60-65% |
| After better data split | 65-75% |
| After transfer learning | 75-85% |
| After ensemble | 85-90% |

## ⚠️ Important Notes

1. **Don't trust validation accuracy alone** - Always test on a separate held-out test set

2. **Medical data requires careful splitting** - Ensure train/val/test are truly independent

3. **Less augmentation can be better** - MRI scans have specific orientations and features

4. **Class imbalance is critical** - Use weighted loss, oversampling, or focal loss

5. **High confidence ≠ Correct** - Your model is overconfident due to overfitting

## 🔍 Root Cause Analysis

The main issue is **NOT the transform mismatch** you were worried about. That was actually correct!

The real issues are:
1. **Overfitting** due to aggressive augmentation
2. **Data split problems** causing train/val similarity
3. **Class imbalance** not properly handled
4. **No test set monitoring** during training

## 📚 Resources

- [Albumentation for medical imaging](https://albumentations.ai/)
- [MedicalNet pre-trained models](https://github.com/Tencent/MedicalNet)
- [Class imbalance in medical imaging](https://arxiv.org/abs/1901.05555)

## 🎓 Lessons Learned

1. ✅ Always validate on a true test set, not just validation
2. ✅ Medical imaging requires domain-specific augmentation
3. ✅ High validation accuracy can be misleading
4. ✅ Monitor train/val/test gap throughout training
5. ✅ Your concern about transforms was valid thinking, even though not the issue!

---

**Next Steps**: Would you like me to create the fixed training notebook with these improvements?
