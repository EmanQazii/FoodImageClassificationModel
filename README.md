# BiteBalance — Food Recognition Model
## Data Science Module | Notebook Documentation

---

## Overview

This notebook covers the complete Data Science pipeline for **BiteBalance**, an AI-based food recognition and dietary tracking mobile application. The goal of this module is to train a deep learning model that can classify food images into 34 categories — including Pakistani foods — and integrate it into a production FastAPI backend.

The notebook is structured in clear phases: dataset preparation, exploratory data analysis, model building with a custom attention layer, multi-round transfer learning, and comprehensive evaluation.

---

## Course Requirements Covered

| Requirement | Implementation |
|---|---|
| Dataset Loading and Exploration | Food-101 + custom Pakistani food classes, ~13,800 images |
| Data Cleaning | Corrupt image removal, size analysis |
| EDA | Class distribution, brightness analysis, sample visualization |
| Feature Engineering | Data augmentation, MobileNetV2 preprocessing |
| Custom Layer | ChannelAttention (Squeeze-and-Excitation inspired) |
| Transfer Learning | MobileNetV2 pretrained on ImageNet |
| Evaluation Metrics | Accuracy, loss, confusion matrix, classification report |
| Result Visualization | Training curves, heatmap, sample predictions grid |

---

## Dataset

- **Source:** Food-101 (Kaggle) + custom curated Pakistani food images
- **Total Images:** ~13,800 after filtering and cleaning
- **Classes:** 34 food categories
- **Split:** 80% training (~11,040 images), 20% validation (2,760 images)
- **Minimum Requirement:** 5,000+ images 

### Food Classes

```
biryani, brownie, butter_chicken, chai, chicken_curry, chicken_tikka,
chicken_wings, chocolate_cake, club_sandwich, cup_cakes, french_fries,
french_toast, fried_rice, garlic_bread, greek_salad, grilled_cheese_sandwich,
haleem, hamburger, hot_and_sour_soup, ice_cream, lasagna, macaroni_and_cheese,
omelette, onion_rings, pancakes, paratha, paratha_roll, pizza,
red_velvet_cake, samosa, spaghetti_carbonara, spring_rolls, steak, waffles
```

> **Note:** Pakistani food classes (biryani, haleem, paratha, paratha\_roll, chai, samosa, chicken\_tikka) were manually curated as they are not present in the original Food-101 dataset.

---

## Project Structure (Google Drive)

```
MyDrive/
├── food_dataset/
│   └── final_food_dataset/
│       ├── biryani/          # ~400 images
│       ├── haleem/           # ~400 images
│       ├── paratha/          # ~400 images
│       └── ...               # 34 class folders total
├── models/
│   ├── history_round1.json
│   ├── history_round2.json
│   ├── history_round3.json
│   ├── best_food_model.h5
│   ├── best_finetuned_v2.h5
│   └── best_finetuned_v3.h5   ← production model
└── food_tracking_model_training.ipynb
```

---

## Notebook Walkthrough

### Step 1 — Setup and Imports

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

Standard data science and deep learning stack. TensorFlow/Keras for model building, OpenCV and PIL for image handling, sklearn for evaluation metrics, matplotlib and seaborn for visualization.

---

### Step 2 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Dataset and saved models are stored on Google Drive. This cell mounts Drive so Colab can access them throughout the notebook.

---

### Step 3 — Data Cleaning

Corrupt images are removed before any analysis or training. A corrupt image in the training pipeline can cause the entire training run to crash.

```python
# iterate through all class folders
for class_name in os.listdir(OUTPUT_DIR):
    class_path = os.path.join(OUTPUT_DIR, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            os.remove(img_path)  # remove corrupt file
```

Image sizes are also analyzed to ensure all images can be resized to 224×224 without significant quality loss.

---

### Step 4 — Exploratory Data Analysis (EDA)

#### Class Distribution

```python
class_counts = {
    cls: len(os.listdir(os.path.join(OUTPUT_DIR, cls)))
    for cls in os.listdir(OUTPUT_DIR)
}

plt.figure(figsize=(16, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Images per Class')
plt.tight_layout()
plt.show()
```

Visualizes how many images exist per class. Important for detecting class imbalance which can cause the model to be biased toward dominant classes.

#### Sample Images

```python
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
for i, class_name in enumerate(list(class_counts.keys())[:32]):
    img_path = os.path.join(OUTPUT_DIR, class_name,
                            os.listdir(os.path.join(OUTPUT_DIR, class_name))[0])
    img = Image.open(img_path)
    axes[i // 8][i % 8].imshow(img)
    axes[i // 8][i % 8].set_title(class_name, fontsize=7)
    axes[i // 8][i % 8].axis('off')
plt.tight_layout()
plt.show()
```

Shows one sample image per class. Confirms dataset quality and visual diversity.

#### Brightness Analysis

```python
brightness_values = []
for class_name in os.listdir(OUTPUT_DIR):
    for img_file in os.listdir(os.path.join(OUTPUT_DIR, class_name)):
        img = cv2.imread(os.path.join(OUTPUT_DIR, class_name, img_file))
        if img is not None:
            brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            brightness_values.append(brightness)

plt.hist(brightness_values, bins=50, color='orange')
plt.title('Image Brightness Distribution')
plt.xlabel('Brightness')
plt.ylabel('Count')
plt.show()
```

Shows the distribution of image brightness across the dataset. A healthy spread indicates diverse lighting conditions which improves model robustness.

---

### Step 5 — Data Augmentation and Generators

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNetV2 preprocessing (-1 to 1)
    rotation_range=20,          # random rotation up to 20 degrees
    zoom_range=0.2,             # random zoom up to 20%
    horizontal_flip=True,       # random horizontal mirror
    width_shift_range=0.1,      # random horizontal shift
    height_shift_range=0.1,     # random vertical shift
    validation_split=0.2        # 80/20 train/val split
)

train_gen = train_datagen.flow_from_directory(
    OUTPUT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    OUTPUT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)
```

**Why augmentation?** Real-world food photos are not perfectly centered or framed. Augmentation artificially creates variations so the model learns to handle different orientations, zooms, and positions. This significantly reduces overfitting.

**Why `preprocess_input`?** MobileNetV2 was originally trained with pixel values in the range -1 to 1. Using the same preprocessing at inference time is critical for correct predictions.

---

### Step 6 — Custom ChannelAttention Layer

This is the custom architectural contribution added on top of MobileNetV2.

```python
import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    """
    Channel Attention Mechanism inspired by Squeeze-and-Excitation Networks.

    Learns to assign importance weights to each feature channel.
    Important channels are amplified, unimportant ones are suppressed.

    Args:
        reduction_ratio (int): Bottleneck reduction factor. Default 8.
    """

    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        # Squeeze: compress spatial dimensions to single value per channel
        self.gap = layers.GlobalAveragePooling2D()

        # Excitation: bottleneck dense layer (channels / reduction_ratio)
        self.dense1 = layers.Dense(
            channels // self.reduction_ratio,
            activation='relu'
        )

        # Excitation: attention weight output (sigmoid → 0 to 1 per channel)
        self.dense2 = layers.Dense(
            channels,
            activation='sigmoid'
        )

        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # Step 1: Squeeze — global average over spatial dimensions
        x = self.gap(inputs)

        # Step 2: Excitation — learn channel importance
        x = self.dense1(x)
        x = self.dense2(x)

        # Step 3: Reshape to (batch, 1, 1, channels) for broadcasting
        x = tf.reshape(x, (-1, 1, 1, inputs.shape[-1]))

        # Step 4: Scale — multiply attention weights with original features
        return inputs * x

    def get_config(self):
        # Required for model save/load with custom layer
        config = super(ChannelAttention, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
```

**How it works:**

```
Input Feature Map (H × W × C)
        ↓
GlobalAveragePooling2D         # Squeeze: (H × W × C) → (C,)
        ↓
Dense(C // reduction_ratio)    # Bottleneck: compress to key features
        ↓
Dense(C, sigmoid)              # Excitation: attention weight per channel
        ↓
Reshape to (1, 1, C)           # Prepare for broadcasting
        ↓
Input × Attention Weights      # Scale: amplify important, suppress unimportant
        ↓
Output Feature Map (H × W × C) # Same shape, reweighted channels
```

**Why this helps for food classification:** Different food items are distinguished by specific visual cues — color channels matter more for identifying biryani vs salad, texture channels matter more for waffles vs pancakes. ChannelAttention lets the model dynamically focus on the most relevant features per input.

> Inspired by: *Squeeze-and-Excitation Networks* (Hu et al., 2018)

---

### Step 7 — Model Architecture

```python
NUM_CLASSES = 34

# Load pretrained MobileNetV2 (ImageNet weights, no top classifier)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers for Round 1 training
base_model.trainable = False

# Build model
inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)   # feature extraction
x = ChannelAttention()(x)               # custom attention layer
x = layers.GlobalAveragePooling2D()(x)  # spatial pooling

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs, name="Food_Classification_Model")
model.summary()
```

**Architecture decisions:**

| Component | Reason |
|---|---|
| MobileNetV2 base | Lightweight, mobile-optimized, strong ImageNet features |
| `include_top=False` | Remove original 1000-class head, add our own 34-class head |
| `base_model.trainable=False` | Preserve pretrained features during initial training |
| ChannelAttention | Custom layer to focus on most relevant feature channels |
| BatchNormalization | Stabilizes training, allows higher learning rates |
| Dropout(0.4 / 0.3) | Prevents overfitting by randomly disabling neurons |
| Softmax output | Produces probability distribution across 34 classes |

---

### Step 8 — Training (3 Rounds)

Training was performed in multiple rounds following a progressive fine-tuning strategy.

#### Round 1 — Top Layers Only

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        '/content/drive/MyDrive/models/best_food_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)
```

Base model frozen. Only the custom head (ChannelAttention + Dense layers) is trained.

#### Round 2 — Partial Fine-tuning

```python
# Unfreeze top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Top 30 layers of MobileNetV2 unfrozen. Lower learning rate to avoid destroying pretrained weights.

#### Round 3 — Full Fine-tuning

```python
# Unfreeze all layers
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # very low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

All layers trainable. Very low learning rate for careful fine-tuning. Best model saved as `best_finetuned_v3.h5`.

**Why multiple rounds?**

Fine-tuning all layers immediately from random initialization would destroy the valuable pretrained features. Progressive unfreezing allows the model to first adapt the classification head, then gradually refine deeper features.

---

### Step 9 — Training History Plots

```python
import json

with open('/content/drive/MyDrive/models/history_round3.json') as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['accuracy'], label='Train Accuracy')
axes[0].plot(history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['loss'], label='Train Loss')
axes[1].plot(history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.show()
```

Training and validation curves confirm the model learned without major overfitting. Train and val curves remain close throughout.

---

### Step 10 — Evaluation

#### Load Best Model

```python
from tensorflow.keras.models import load_model

model = load_model(
    '/content/drive/MyDrive/models/best_finetuned_v3.h5',
    custom_objects={'ChannelAttention': ChannelAttention}
)
```

> `custom_objects` must be passed so Keras can reconstruct the ChannelAttention layer on load.

#### Run Predictions

```python
val_gen.reset()
predictions = model.predict(val_gen, verbose=1)
pred_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes
class_names = list(val_gen.class_indices.keys())
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

#### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(true_classes, pred_classes, target_names=class_names))
```

---

### Step 11 — Sample Predictions Visualization

```python
val_gen.reset()
batch_images, batch_labels = next(val_gen)

fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.flatten()

for i in range(20):
    img = batch_images[i]
    true_label = class_names[np.argmax(batch_labels[i])]

    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_label = class_names[np.argmax(pred)]

    # Reverse MobileNetV2 preprocessing for display
    display_img = img.copy()
    display_img /= 2.0
    display_img += 0.5
    display_img = np.clip(display_img, 0, 1)

    axes[i].imshow(display_img)
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle('Sample Predictions (Green = Correct, Red = Wrong)')
plt.tight_layout()
plt.show()
```

Shows 20 random validation images with true and predicted labels. Green title = correct, red title = wrong. Gives a qualitative sense of model performance on real images.

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | **76.49%** |
| Macro Avg F1 | **0.78** |
| Weighted Avg F1 | **0.77** |
| Test Samples | 2,760 |
| Total Classes | 34 |

### Top Performing Classes

| Class | F1 Score |
|---|---|
| spaghetti_carbonara | 0.93 |
| brownie | 0.93 |
| hot_and_sour_soup | 0.92 |
| chai | 0.89 |
| greek_salad | 0.89 |
| paratha | 0.89 |
| french_fries | 0.89 |
| paratha_roll | 0.86 |
| onion_rings | 0.86 |

### Pakistani Food Classes Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| biryani | 0.65 | 0.85 | 0.74 |
| haleem | 0.71 | 0.75 | 0.73 |
| paratha | 0.94 | 0.85 | 0.89 |
| paratha_roll | 0.79 | 0.95 | 0.86 |
| chai | 0.80 | 1.00 | 0.89 |
| samosa | 0.72 | 0.77 | 0.74 |
| chicken_tikka | 0.74 | 0.85 | 0.79 |

---

## Model Integration

The trained model `best_finetuned_v3.h5` is loaded in the FastAPI backend:

```python
# backend/services/ml_model.py

from tensorflow.keras.models import load_model
from ml.custom_layers import ChannelAttention

model = load_model(
    'ml/saved_model/best_finetuned_v3.h5',
    custom_objects={'ChannelAttention': ChannelAttention}
)
```

Prediction pipeline:

```
User uploads image (Flutter app)
        ↓
FastAPI receives image bytes
        ↓
DIP Pipeline (OpenCV preprocessing)
  - resize to 224×224
  - normalize
  - denoise
        ↓
MobileNetV2 + ChannelAttention model
        ↓
Softmax output (34 class probabilities)
        ↓
Top class + confidence score returned
        ↓
Calorie mapping (hardcoded ranges, USDA verified)
        ↓
Result displayed to user
```

---

## Dependencies

```
tensorflow>=2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
```

Install in Colab:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow
```

---

## References

- Howard, A. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
- Hu, J. et al. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.
- Bossard, L. et al. (2014). *Food-101 — Mining Discriminative Components with Random Forests*. ECCV 2014.
- USDA FoodData Central. https://fdc.nal.usda.gov

---

*BiteBalance — AI Food Recognition and Dietary Tracking System*
*Data Science Module | Semester Project*
