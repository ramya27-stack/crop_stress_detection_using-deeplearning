# crop_stress_detection_using-deeplearning# 🌿 Multicrop Stress Detection & Traditional Recommendations System
**Framework:** TensorFlow / Keras  
**Model:** EfficientNetB3 (Transfer Learning)  
**Dataset:** Kaggle (PlantVillage / New Plant Diseases Dataset)  
**Crops:** Rice, Wheat, Maize, Tomato, Potato, Cotton, Banana, Sugarcane, Ash Gourd, Cherry  

> This notebook covers: Installation → Data Loading → Preprocessing → Model Training → Evaluation → Recommendations
## 📦 Step 1: Install Dependencies
# Install required packages
!pip install tensorflow keras scikit-learn matplotlib seaborn Pillow -q
print('✅ All packages installed')
## 📚 Step 2: Import Libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix

print(f'✅ TensorFlow version: {tf.__version__}')
print(f'✅ GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
print(f'✅ GPUs: {tf.config.list_physical_devices("GPU")}')
## 📥 Step 3: Download Kaggle Dataset

**Option A** — Use Kaggle API (recommended in Colab/Kaggle Notebook)  
**Option B** — Manual: Upload your dataset folder and set `DATA_DIR` below
# ── OPTION A: Kaggle API ──────────────────────────────────────────────
# Uncomment and run if using Google Colab or Kaggle Notebook

# from google.colab import files
# files.upload()  # Upload your kaggle.json API key

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Download the New Plant Diseases dataset
# !kaggle datasets download -d vipoooool/new-plant-diseases-dataset
# !unzip new-plant-diseases-dataset.zip -d data/ -q
# print('✅ Dataset downloaded and extracted')

# ── OPTION B: Set your local data path ───────────────────────────────
DATA_DIR = './data'          # Change this to your dataset path
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')

# Check what classes exist
if os.path.exists(TRAIN_DIR):
    classes = sorted(os.listdir(TRAIN_DIR))
    print(f'✅ Found {len(classes)} classes in training set')
    print('   First 10:', classes[:10])
else:
    print('⚠️  data/train/ not found. Please set DATA_DIR above.')
    print('   Expected structure:')
    print('   data/train/Rice_LeafBlight/*.jpg')
    print('   data/train/Tomato_EarlyBlight/*.jpg')
    print('   data/test/Rice_LeafBlight/*.jpg  ...')
## ⚙️ Step 4: Configuration
# ─── GLOBAL CONFIG ───────────────────────────────────────────────────
CONFIG = {
    'img_size'        : (224, 224),
    'batch_size'      : 32,
    'epochs'          : 50,
    'learning_rate'   : 1e-4,
    'dropout_rate'    : 0.4,
    'data_dir'        : './data',
    'model_save_path' : './saved_model/multicrop_stress_model.h5',
    'results_dir'     : './results',
    'seed'            : 42,
}

os.makedirs('./saved_model', exist_ok=True)
os.makedirs('./results', exist_ok=True)

tf.random.set_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print('✅ Configuration set:')
for k, v in CONFIG.items():
    print(f'   {k:<20} = {v}')
## 🔄 Step 5: Data Preprocessing & Augmentation
def create_data_generators(data_dir):
    """Create train/val/test generators with augmentation."""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=CONFIG['seed']
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=CONFIG['seed']
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


train_gen, val_gen, test_gen = create_data_generators(CONFIG['data_dir'])
num_classes = len(train_gen.class_indices)

# Save class index mapping
with open(f"{CONFIG['results_dir']}/class_indices.json", 'w') as f:
    json.dump(train_gen.class_indices, f, indent=2)

print(f'✅ Number of classes : {num_classes}')
print(f'   Training samples  : {train_gen.samples}')
print(f'   Validation samples: {val_gen.samples}')
print(f'   Test samples      : {test_gen.samples}')
## 🖼️ Step 6: Visualise Sample Images
def show_sample_images(generator, n=12):
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    images, labels = next(generator)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0f0a')
    fig.suptitle('Sample Training Images', color='white', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            label = idx_to_class[np.argmax(labels[i])]
            crop, *stress = label.split('_')
            ax.set_title(f'{crop}\n{"_".join(stress)}', color='#4ade80', fontsize=8)
        ax.axis('off')
        ax.set_facecolor('#111811')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/sample_images.png", dpi=120, bbox_inches='tight')
    plt.show()
    print('✅ Sample images displayed')

show_sample_images(train_gen)
## 📊 Step 7: Class Distribution
def plot_class_distribution(generator):
    labels = generator.classes
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    counts = np.bincount(labels)

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5), 6))
    fig.patch.set_facecolor('#0a0f0a')
    ax.set_facecolor('#111811')

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(class_names)))
    bars = ax.bar(range(len(class_names)), counts, color=colors, edgecolor='#1e2e1e')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', color='#aaaaaa', fontsize=8)
    ax.set_ylabel('Number of Images', color='#5a7a5a')
    ax.set_title('Class Distribution (Training Set)', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#5a7a5a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2e1e')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/class_distribution.png", dpi=120, bbox_inches='tight',
                facecolor='#0a0f0a')
    plt.show()

plot_class_distribution(train_gen)
## 🏗️ Step 8: Build EfficientNetB3 Model
def build_model(num_classes):
    """
    EfficientNetB3 backbone + custom classification head.
    Multi-class output for all crop × stress combinations.
    """
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['img_size'], 3)
    )
    base_model.trainable = False  # Freeze initially

    inputs  = keras.Input(shape=(*CONFIG['img_size'], 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(512, activation='relu')(x)
    x       = layers.Dropout(CONFIG['dropout_rate'])(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(CONFIG['dropout_rate'] * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='MulticropStressDetector')
    return model, base_model


model, base_model = build_model(num_classes)

model.compile(
    optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
    ]
)

model.summary()
print(f'\n✅ Model built: {model.count_params():,} total parameters')
print(f'   Trainable  : {sum(tf.size(v).numpy() for v in model.trainable_variables):,}')
## ⚡ Step 9: Set Up Training Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        CONFIG['model_save_path'],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
]
print('✅ Callbacks configured:')
print('   • EarlyStopping      (patience=8)')
print('   • ModelCheckpoint    (saves best val_accuracy)')
print('   • ReduceLROnPlateau  (factor=0.3, patience=4)')
## 🚀 Step 10: Phase 1 – Train Classification Head
> EfficientNetB3 base is **frozen**. Only the custom head layers train.
print('🚀 Phase 1: Training classification head (base frozen)...')
print(f'   Epochs  : {CONFIG["epochs"]}')
print(f'   Batch   : {CONFIG["batch_size"]}')
print(f'   LR      : {CONFIG["learning_rate"]}')
print()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=CONFIG['epochs'],
    callbacks=callbacks,
    verbose=1
)

best_val_acc = max(history.history['val_accuracy'])
print(f'\n✅ Phase 1 complete. Best val accuracy: {best_val_acc*100:.2f}%')
## 🔓 Step 11: Phase 2 – Fine-Tuning
> Unfreeze the **top 30 layers** of EfficientNetB3 and train with a lower learning rate.
print('🔓 Phase 2: Fine-tuning top 30 layers...')

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

fine_tune_lr = CONFIG['learning_rate'] / 10
model.compile(
    optimizer=keras.optimizers.Adam(fine_tune_lr),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
    ]
)

trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f'   Fine-tune LR    : {fine_tune_lr}')
print(f'   Trainable params: {trainable_count:,}')
print()

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

best_fine_acc = max(history_fine.history['val_accuracy'])
print(f'\n✅ Phase 2 complete. Best val accuracy: {best_fine_acc*100:.2f}%')
## 📈 Step 12: Plot Training History
def plot_training_history(h1, h2=None):
    acc     = h1.history['accuracy']
    val_acc = h1.history['val_accuracy']
    loss    = h1.history['loss']
    val_loss= h1.history['val_loss']

    if h2:
        acc      += h2.history['accuracy']
        val_acc  += h2.history['val_accuracy']
        loss     += h2.history['loss']
        val_loss += h2.history['val_loss']
        split     = len(h1.history['accuracy'])
    else:
        split = None

    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0f0a')

    for ax in axes:
        ax.set_facecolor('#111811')
        ax.tick_params(colors='#5a7a5a')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2e1e')
        if split:
            ax.axvline(x=split, color='#fbbf24', linestyle='--', alpha=0.5, label='Fine-tune start')

    axes[0].plot(epochs_range, acc,     color='#4ade80', lw=2, label='Train Accuracy')
    axes[0].plot(epochs_range, val_acc, color='#60a5fa', lw=2, label='Val Accuracy', ls='--')
    axes[0].set_title('Model Accuracy', color='white', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', color='#5a7a5a')
    axes[0].set_ylabel('Accuracy', color='#5a7a5a')
    axes[0].legend(facecolor='#111811', labelcolor='white')
    axes[0].set_ylim([0, 1])

    axes[1].plot(epochs_range, loss,     color='#f87171', lw=2, label='Train Loss')
    axes[1].plot(epochs_range, val_loss, color='#fbbf24', lw=2, label='Val Loss', ls='--')
    axes[1].set_title('Model Loss', color='white', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', color='#5a7a5a')
    axes[1].set_ylabel('Loss', color='#5a7a5a')
    axes[1].legend(facecolor='#111811', labelcolor='white')

    plt.suptitle('Multicrop Stress Detection — Training History', color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/training_history.png", dpi=150,
                bbox_inches='tight', facecolor='#0a0f0a')
    plt.show()
    print('✅ Training history saved.')

plot_training_history(history, history_fine)
## 🧪 Step 13: Evaluate on Test Set
print('📊 Evaluating on test set...')
results = model.evaluate(test_gen, verbose=1)

print(f'\n' + '='*50)
print(f'  ✅ Test Loss         : {results[0]:.4f}')
print(f'  ✅ Test Accuracy     : {results[1]*100:.2f}%')
print(f'  ✅ Top-3 Accuracy    : {results[2]*100:.2f}%')
print('='*50)
## 📋 Step 14: Per-Class Classification Report
# Predict on test set
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes
class_names  = list(test_gen.class_indices.keys())

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))

# Save report
with open(f"{CONFIG['results_dir']}/classification_report.json", 'w') as f:
    json.dump(report, f, indent=2)
print('✅ Classification report saved to results/classification_report.json')
## 🔢 Step 15: Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(12, n), max(10, n - 2)))
    fig.patch.set_facecolor('#0a0f0a')
    ax.set_facecolor('#111811')

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Greens',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor='#1e2e1e',
        annot_kws={'size': 7}
    )
    ax.set_title('Confusion Matrix (Normalized)', color='white', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', color='#5a7a5a', fontsize=10)
    ax.set_ylabel('True Label', color='#5a7a5a', fontsize=10)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/confusion_matrix.png", dpi=150,
                bbox_inches='tight', facecolor='#0a0f0a')
    plt.show()
    print('✅ Confusion matrix saved.')

plot_confusion_matrix(y_true, y_pred, class_names)
## 📊 Step 16: Per-Class F1 Score Chart
f1_scores  = [report[c]['f1-score'] for c in class_names if c in report]
precisions = [report[c]['precision'] for c in class_names if c in report]
recalls    = [report[c]['recall'] for c in class_names if c in report]

x = np.arange(len(class_names))
w = 0.25

fig, ax = plt.subplots(figsize=(max(14, len(class_names) * 0.6), 6))
fig.patch.set_facecolor('#0a0f0a')
ax.set_facecolor('#111811')

ax.bar(x - w, precisions, w, label='Precision', color='#60a5fa', alpha=0.85)
ax.bar(x,     f1_scores,  w, label='F1-Score',  color='#4ade80', alpha=0.85)
ax.bar(x + w, recalls,    w, label='Recall',    color='#fbbf24', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right', color='#aaaaaa', fontsize=8)
ax.set_ylim([0, 1.1])
ax.set_ylabel('Score', color='#5a7a5a')
ax.set_title('Per-Class Precision / F1 / Recall', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='#5a7a5a')
ax.legend(facecolor='#111811', labelcolor='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#1e2e1e')

plt.tight_layout()
plt.savefig(f"{CONFIG['results_dir']}/f1_scores.png", dpi=150,
            bbox_inches='tight', facecolor='#0a0f0a')
plt.show()
print('✅ F1 score chart saved.')
## 🌱 Step 17: Traditional Recommendations & Fertilizer System
# ─── RECOMMENDATION DATABASE ────────────────────────────────────────────────
RECOMMENDATIONS = {

    # ── RICE ──────────────────────────────────────────────────────────────────
    'Rice_LeafBlight': {
        'crop': 'Rice', 'stress': 'Bacterial Leaf Blight', 'category': 'Biotic – Bacterial',
        'severity': 'Yellow water-soaked lesions on leaf edges turning brown',
        'remedies': [
            'Spray neem leaf extract (1kg boiled in 5L water, diluted 1:10) every 7 days',
            'Apply wood ash around plant base to reduce soil moisture',
            'Remove and burn infected leaves immediately',
            'Avoid waterlogging — ensure proper field drainage',
        ],
        'organic':  {'name': 'Compost + Wood Ash', 'dose': '2–3 tons/acre compost + 20 kg/acre wood ash',
                     'reason': 'Potassium strengthens cell walls, reducing bacterial entry', 'timing': 'At tillering stage'},
        'chemical': {'name': 'Potassium Chloride (MOP) + Copper Oxychloride',
                     'dose': '60 kg/acre MOP; 0.3% Copper Oxychloride spray',
                     'reason': 'Copper acts as bactericide; K improves resistance', 'timing': 'Foliar spray at first symptom'},
        'prevention': 'Use resistant varieties (IR64, Swarna); avoid high N at early stages'
    },
    'Rice_BlastDisease': {
        'crop': 'Rice', 'stress': 'Rice Blast (Magnaporthe oryzae)', 'category': 'Biotic – Fungal',
        'severity': 'Diamond-shaped gray lesions with brown border on leaves',
        'remedies': [
            'Spray garlic extract (100g crushed in 1L water, diluted 1:5)',
            'Apply silicon-rich rice hull ash to soil surface',
            'Intercrop with marigold as trap crop',
            'Reduce nitrogen application — excess N worsens blast',
        ],
        'organic':  {'name': 'Rice Hull Ash + Neem Cake', 'dose': '500 kg/acre rice hull ash + 100 kg/acre neem cake',
                     'reason': 'Silicon deposits in cell walls block fungal penetration', 'timing': 'Basal before transplanting'},
        'chemical': {'name': 'Tricyclazole 75% WP', 'dose': '0.6 g/L foliar spray; NPK 120:60:60',
                     'reason': 'Tricyclazole is the standard blast fungicide', 'timing': 'Spray at boot and heading stage'},
        'prevention': 'Avoid dense planting; maintain field sanitation'
    },

    # ── WHEAT ─────────────────────────────────────────────────────────────────
    'Wheat_YellowRust': {
        'crop': 'Wheat', 'stress': 'Yellow Rust (Stripe Rust)', 'category': 'Biotic – Fungal',
        'severity': 'Yellow-orange pustules in stripes along leaf veins',
        'remedies': [
            'Spray cow urine solution (1:10 dilution) as antifungal',
            'Dust sulfur powder on leaves in early morning',
            'Apply turmeric powder in water (50g/L) as foliar spray',
        ],
        'organic':  {'name': 'Vermicompost + Potassium Silicate',
                     'dose': '1 ton/acre vermicompost + 2% potassium silicate foliar',
                     'reason': 'Silicon and micronutrients boost immunity', 'timing': 'Crown root initiation'},
        'chemical': {'name': 'Propiconazole 25% EC (Tilt)',
                     'dose': '0.1% foliar spray (1 mL/L); NPK 120:60:40',
                     'reason': 'Propiconazole is highly effective against rust fungi',
                     'timing': 'Spray at first rust; repeat after 15 days'},
        'prevention': 'Sow resistant varieties; early sowing avoids peak rust season'
    },

    # ── MAIZE ─────────────────────────────────────────────────────────────────
    'Maize_CommonRust': {
        'crop': 'Maize', 'stress': 'Common Rust', 'category': 'Biotic – Fungal',
        'severity': 'Brick-red oval pustules on both leaf surfaces',
        'remedies': [
            'Dust fine sulfur powder at 10 kg/acre in early morning',
            'Spray onion-garlic extract (blend 100g each, dilute 1:8)',
            'Remove lower infected leaves to improve air circulation',
        ],
        'organic':  {'name': 'Compost Tea + Seaweed Extract',
                     'dose': '5L/acre compost tea; 2 mL/L seaweed foliar',
                     'reason': 'Seaweed activates plant defense enzymes (SAR)', 'timing': 'V6–V10 stage'},
        'chemical': {'name': 'Propiconazole 25% EC + MOP',
                     'dose': '1 mL/L foliar spray; 60 kg/acre MOP',
                     'reason': 'Propiconazole controls rust; K reduces susceptibility',
                     'timing': 'Spray at first pustule appearance'},
        'prevention': 'Plant before monsoon to avoid peak spore load'
    },
    'Maize_NorthernLeafBlight': {
        'crop': 'Maize', 'stress': 'Northern Leaf Blight', 'category': 'Biotic – Fungal',
        'severity': 'Long gray-green to tan cigar-shaped lesions on leaves',
        'remedies': [
            'Spray fermented buttermilk (1:5 dilution) as foliar spray',
            'Apply Trichoderma viride bioagent at root zone',
            'Remove infected crop debris after harvest',
        ],
        'organic':  {'name': 'FYM + Zinc Sulfate',
                     'dose': '5 tons/acre FYM + 25 kg/acre zinc sulfate',
                     'reason': 'Zinc improves disease tolerance and grain filling', 'timing': 'Basal at sowing'},
        'chemical': {'name': 'Azoxystrobin + NPK 180:90:90',
                     'dose': '1 mL/L foliar azoxystrobin; NPK split doses',
                     'reason': 'Azoxystrobin (strobilurin) highly effective against leaf blights',
                     'timing': 'Spray at V8–V10 stage'},
        'prevention': 'Resistant hybrids; avoid dense canopy planting'
    },

    # ── TOMATO ────────────────────────────────────────────────────────────────
    'Tomato_EarlyBlight': {
        'crop': 'Tomato', 'stress': 'Early Blight (Alternaria)', 'category': 'Biotic – Fungal',
        'severity': 'Concentric ring bull\'s-eye lesions on lower leaves',
        'remedies': [
            'Spray baking soda solution (1 tsp/L water + few drops soap)',
            'Apply Bordeaux mixture (1% copper solution)',
            'Mulch base of plants with straw to prevent soil splash',
            'Remove lower leaves touching soil surface',
        ],
        'organic':  {'name': 'Vermicompost + Calcium Nitrate',
                     'dose': '2 tons/acre vermicompost; 5g/L calcium nitrate foliar',
                     'reason': 'Calcium strengthens cell walls; reduces Alternaria',
                     'timing': 'Calcium spray every 2 weeks'},
        'chemical': {'name': 'Mancozeb 75% WP + 19:19:19 NPK',
                     'dose': '2.5 g/L Mancozeb; 5 g/L 19:19:19 water-soluble',
                     'reason': 'Mancozeb is the standard early blight fungicide',
                     'timing': 'Weekly spray from flowering stage'},
        'prevention': 'Avoid overhead irrigation; stake plants for air circulation'
    },
    'Tomato_LateBlight': {
        'crop': 'Tomato', 'stress': 'Late Blight (Phytophthora)', 'category': 'Biotic – Oomycete',
        'severity': 'Water-soaked dark lesions on leaves/fruits; white mold under leaves',
        'remedies': [
            'Spray Bordeaux mixture (1 kg blue vitriol + 1 kg lime in 100L)',
            'Dust wood ash on foliage in early morning',
            'Improve drainage; avoid water stagnation',
        ],
        'organic':  {'name': 'Neem Cake + Potassium Humate',
                     'dose': '300 kg/acre neem cake; 5 kg/acre potassium humate',
                     'reason': 'Potassium humate activates soil suppressive microbes',
                     'timing': 'At transplanting and 30 days after'},
        'chemical': {'name': 'Metalaxyl + Mancozeb (Ridomil Gold)',
                     'dose': '2.5 g/L spray every 7–10 days',
                     'reason': 'Metalaxyl systemic; highly effective against Phytophthora',
                     'timing': 'Preventive before rainy season; curative at first symptoms'},
        'prevention': 'Never compost infected plants; rotate with non-solanaceous crops'
    },

    # ── POTATO ────────────────────────────────────────────────────────────────
    'Potato_EarlyBlight': {
        'crop': 'Potato', 'stress': 'Early Blight (Alternaria solani)', 'category': 'Biotic – Fungal',
        'severity': 'Dark brown concentric ring lesions on older leaves',
        'remedies': [
            'Spray Bordeaux mixture (0.5%) every 10 days',
            'Apply turmeric + lime powder dust on leaves',
            'Ensure proper hilling to avoid soil splash on leaves',
        ],
        'organic':  {'name': 'FYM + Calcium Superphosphate',
                     'dose': '10 tons/acre FYM + 150 kg/acre SSP',
                     'reason': 'Calcium strengthens foliage; P for root health', 'timing': 'Basal at planting'},
        'chemical': {'name': 'Chlorothalonil 75% WP',
                     'dose': '2 g/L spray every 7–10 days; NPK 180:90:120',
                     'reason': 'Chlorothalonil is standard early blight fungicide for potato',
                     'timing': 'Start at vine closure; until harvest – 30 days'},
        'prevention': 'Use certified disease-free seed; avoid late planting'
    },
    'Potato_LateBlight': {
        'crop': 'Potato', 'stress': 'Late Blight (Phytophthora infestans)', 'category': 'Biotic – Oomycete',
        'severity': 'Dark water-soaked lesions; white sporangia on leaf undersides',
        'remedies': [
            'Spray Bordeaux mixture (1%) at 7-day intervals',
            'Apply copper oxychloride dust in early morning',
            'Hill up soil around plants to protect tubers',
        ],
        'organic':  {'name': 'Compost + Potassium Silicate',
                     'dose': '5 tons/acre compost; 2% potassium silicate foliar',
                     'reason': 'Silicon proven to reduce late blight severity',
                     'timing': 'Foliar from 45 days after planting'},
        'chemical': {'name': 'Cymoxanil + Mancozeb (Curzate M8)',
                     'dose': '2.5 g/L spray every 5–7 days in wet weather',
                     'reason': 'Cymoxanil curative; Mancozeb protective',
                     'timing': 'Preventive sprays at haulm emergence'},
        'prevention': 'Destroy volunteer plants; do not plant in infected soil'
    },

    # ── COTTON ────────────────────────────────────────────────────────────────
    'Cotton_BollWeevil': {
        'crop': 'Cotton', 'stress': 'Boll Weevil Infestation', 'category': 'Biotic – Insect Pest',
        'severity': 'Punctured squares/bolls with entry holes; premature shedding',
        'remedies': [
            'Pheromone traps (5/acre) for monitoring and mass trapping',
            'Spray neem seed kernel extract (NSKE) 5%',
            'Apply tobacco decoction spray (1 kg/10L, diluted 1:5)',
            'Hand-pick and destroy affected bolls',
        ],
        'organic':  {'name': 'Neem Cake + FYM',
                     'dose': '500 kg/acre neem cake + 5 tons/acre FYM',
                     'reason': 'Neem acts as soil insecticide reducing larval population',
                     'timing': 'Basal at planting'},
        'chemical': {'name': 'Malathion 50 EC + NPK 120:60:60',
                     'dose': '2 mL/L malathion foliar; NPK in split doses',
                     'reason': 'Malathion is standard contact insecticide for boll weevil',
                     'timing': 'Spray at square formation; repeat every 15 days'},
        'prevention': 'Early sowing; destroy crop residues after harvest'
    },

    # ── BANANA ────────────────────────────────────────────────────────────────
    'Banana_PanamaWilt': {
        'crop': 'Banana', 'stress': 'Panama Wilt (Fusarium Wilt)', 'category': 'Biotic – Fungal',
        'severity': 'Yellowing from outer leaves inward; brown vascular discoloration in pseudostem',
        'remedies': [
            'Apply Trichoderma viride (10g/plant at root zone)',
            'Drench soil with cow dung slurry + lime (1:1) monthly',
            'Remove and burn infected plants with their corms',
        ],
        'organic':  {'name': 'Trichoderma-enriched Compost + Wood Ash',
                     'dose': '5 kg/plant Trichoderma compost; 500g/plant wood ash',
                     'reason': 'Trichoderma competes with Fusarium in soil',
                     'timing': 'At planting and every 3 months'},
        'chemical': {'name': 'Carbendazim soil drench + NPK 200:60:200',
                     'dose': '2 g/L carbendazim root drench; high K for wilt resistance',
                     'reason': 'High K:N ratio reduces Fusarium susceptibility',
                     'timing': 'Drench at planting; NPK split over crop duration'},
        'prevention': 'Use disease-free tissue culture planting material'
    },

    # ── SUGARCANE ─────────────────────────────────────────────────────────────
    'Sugarcane_RedRot': {
        'crop': 'Sugarcane', 'stress': 'Red Rot (Colletotrichum falcatum)', 'category': 'Biotic – Fungal',
        'severity': 'Red discoloration inside stalk with white patches; sour smell',
        'remedies': [
            'Soak seed setts in Trichoderma viride (250g/100L) for 30 min',
            'Remove and burn infected stools',
            'Apply neem cake in furrows at planting',
        ],
        'organic':  {'name': 'Trichoderma-enriched FYM + SSP',
                     'dose': '5 tons/acre T.viride FYM; 80 kg/acre SSP',
                     'reason': 'Trichoderma suppresses Colletotrichum in soil', 'timing': 'Basal at planting'},
        'chemical': {'name': 'Carbendazim sett treatment + NPK 250:80:120',
                     'dose': '1 g/L carbendazim sett dip for 30 min; split NPK',
                     'reason': 'Systemic fungicide protects young setts',
                     'timing': 'Sett treatment before planting'},
        'prevention': 'Use disease-free planting material; resistant varieties (Co 86032)'
    },

    # ── ASH GOURD ─────────────────────────────────────────────────────────────
    'AshGourd_PowderyMildew': {
        'crop': 'Ash Gourd', 'stress': 'Powdery Mildew', 'category': 'Biotic – Fungal',
        'severity': 'White floury patches on upper leaf surface; yellowing',
        'remedies': [
            'Spray buttermilk diluted 1:10 with water every 5 days',
            'Apply neem oil + baking soda (3 mL neem + 1 tsp soda per L)',
            'Dust sulfur powder 8–10 kg/acre in early morning',
        ],
        'organic':  {'name': 'Vermicompost + Potassium Silicate',
                     'dose': '2 tons/acre + 5 kg/acre potassium silicate',
                     'reason': 'Silicon strengthens epidermal cells against mildew',
                     'timing': 'Basal + foliar at 30 and 50 DAS'},
        'chemical': {'name': 'Hexaconazole 5% SC + NPK 80:60:60',
                     'dose': '1 mL/L hexaconazole; 2–3 sprays at 10-day interval',
                     'reason': 'Hexaconazole is systemic DMI fungicide for cucurbit mildew',
                     'timing': 'Spray at first symptom; preventive from flowering'},
        'prevention': 'Train vines on trellis; avoid dense planting'
    },

    # ── CHERRY ────────────────────────────────────────────────────────────────
    'Cherry_PowderyMildew': {
        'crop': 'Cherry', 'stress': 'Powdery Mildew', 'category': 'Biotic – Fungal',
        'severity': 'White powdery coating on young leaves, shoots, and fruit',
        'remedies': [
            'Spray dilute milk (1:9 milk:water) — proteins act as antifungal',
            'Apply baking soda solution (1 tbsp/L + few drops oil)',
            'Dust fine sulfur powder in early morning before dew dries',
            'Prune to improve air circulation through canopy',
        ],
        'organic':  {'name': 'Compost + Calcium Foliar',
                     'dose': '5 kg/tree compost; 5g/L calcium chloride foliar',
                     'reason': 'Calcium hardens fruit skin; reduces mildew entry',
                     'timing': 'Foliar at petal fall and 2–3 times after'},
        'chemical': {'name': 'Myclobutanil 10% WP + NPK 100:50:80',
                     'dose': '1 g/L myclobutanil spray every 10–14 days',
                     'reason': 'Myclobutanil (DMI fungicide) highly effective against powdery mildew',
                     'timing': 'First spray at bud burst; continue through fruit development'},
        'prevention': 'Prune dense canopies; avoid high nitrogen; plant resistant cultivars'
    },
}

print(f'✅ Recommendation engine loaded: {len(RECOMMENDATIONS)} crop-stress conditions')
print('   Crops covered:', sorted(set(v["crop"] for v in RECOMMENDATIONS.values())))
## 🔍 Step 18: Recommendation Lookup Function
def get_recommendation(label: str) -> dict:
    """Return recommendation for a predicted label."""
    if label in RECOMMENDATIONS:
        return RECOMMENDATIONS[label]
    for key, val in RECOMMENDATIONS.items():
        if label.lower().replace(' ', '_') in key.lower():
            return val
    return {'crop': label, 'stress': 'Unknown', 'category': 'Unknown',
            'severity': 'Consult local agricultural extension officer',
            'remedies': ['Consult local agricultural extension officer'],
            'organic': {'name': 'Balanced compost', 'dose': 'As per soil test',
                        'reason': 'Soil health', 'timing': 'Pre-season'},
            'chemical': {'name': 'NPK as per soil test', 'dose': 'Per crop recommendation',
                         'reason': 'Balanced nutrition', 'timing': 'Split doses'},
            'prevention': 'Regular scouting; consult extension officer'}


def print_recommendation(label: str):
    """Pretty print recommendation for a given label."""
    rec = get_recommendation(label)
    print(f'\n{"="*60}')
    print(f'  🌿 CROP          : {rec["crop"]}')
    print(f'  🔬 STRESS        : {rec["stress"]}')
    print(f'  🏷  CATEGORY     : {rec["category"]}')
    print(f'  🔍 IDENTIFY BY   : {rec["severity"]}')
    print(f'\n  🌱 TRADITIONAL REMEDIES:')
    for r in rec['remedies']:
        print(f'     • {r}')
    org = rec['organic']
    print(f'\n  🧺 ORGANIC FERTILIZER:')
    print(f'     Name   : {org["name"]}')
    print(f'     Dose   : {org["dose"]}')
    print(f'     Reason : {org["reason"]}')
    print(f'     Timing : {org["timing"]}')
    chem = rec['chemical']
    print(f'\n  💊 CHEMICAL FERTILIZER / PESTICIDE:')
    print(f'     Name   : {chem["name"]}')
    print(f'     Dose   : {chem["dose"]}')
    print(f'     Reason : {chem["reason"]}')
    print(f'     Timing : {chem["timing"]}')
    print(f'\n  🛡  PREVENTION   : {rec["prevention"]}')
    print('='*60)


# Example: look up recommendations for Tomato Early Blight
print_recommendation('Tomato_EarlyBlight')
## 🔮 Step 19: Predict a Single Image + Get Recommendations
from tensorflow.keras.preprocessing import image as keras_image

def predict_and_recommend(img_path: str, top_k: int = 3):
    """
    Full pipeline: load image → predict → print recommendation.
    """
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}

    # Load and preprocess
    img      = keras_image.load_img(img_path, target_size=CONFIG['img_size'])
    img_arr  = keras_image.img_to_array(img) / 255.0
    img_arr  = np.expand_dims(img_arr, axis=0)

    # Predict
    preds      = model.predict(img_arr, verbose=0)[0]
    top_idx    = np.argsort(preds)[::-1][:top_k]
    top_label  = idx_to_class[top_idx[0]]
    top_conf   = preds[top_idx[0]]

    # Display image
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0f0a')

    axes[0].imshow(img)
    axes[0].set_title(f'Input Image', color='white')
    axes[0].axis('off')
    axes[0].set_facecolor('#111811')

    # Confidence bar chart
    labels = [idx_to_class[i] for i in top_idx]
    confs  = [preds[i] for i in top_idx]
    colors_bar = ['#4ade80', '#60a5fa', '#fbbf24'][:len(labels)]
    axes[1].set_facecolor('#111811')
    bars = axes[1].barh(range(len(labels)), confs, color=colors_bar)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, color='#aaaaaa', fontsize=9)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Confidence', color='#5a7a5a')
    axes[1].set_title(f'Top-{top_k} Predictions', color='white', fontweight='bold')
    axes[1].tick_params(colors='#5a7a5a')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#1e2e1e')
    for bar, conf in zip(bars, confs):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{conf*100:.1f}%', va='center', color='white', fontsize=9)

    plt.suptitle(f'Prediction: {top_label}  ({top_conf*100:.1f}%)',
                 color='#4ade80', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print recommendation
    print_recommendation(top_label)


# Usage:
# predict_and_recommend('./data/test/Tomato_EarlyBlight/image001.jpg')
print('✅ predict_and_recommend() function ready.')
print('   Usage: predict_and_recommend("path/to/leaf_image.jpg")')
## 📊 Step 20: Accuracy Summary Table
# Build a summary DataFrame from classification report
summary_rows = []
for cls in class_names:
    if cls in report and cls not in ['accuracy', 'macro avg', 'weighted avg']:
        crop_stress = cls.split('_', 1)
        crop   = crop_stress[0]
        stress = crop_stress[1] if len(crop_stress) > 1 else cls
        summary_rows.append({
            'Crop'     : crop,
            'Stress'   : stress,
            'Precision': f"{report[cls]['precision']*100:.1f}%",
            'Recall'   : f"{report[cls]['recall']*100:.1f}%",
            'F1-Score' : f"{report[cls]['f1-score']*100:.1f}%",
            'Support'  : int(report[cls]['support'])
        })

df_results = pd.DataFrame(summary_rows)
print('\n📋 Per-Class Accuracy Results:')
print('='*70)
print(df_results.to_string(index=False))
print('='*70)
print(f'\n  Overall Accuracy  : {results[1]*100:.2f}%')
print(f'  Top-3 Accuracy    : {results[2]*100:.2f}%')
print(f'  Macro Avg F1      : {report["macro avg"]["f1-score"]*100:.2f}%')

df_results.to_csv(f"{CONFIG['results_dir']}/accuracy_summary.csv", index=False)
print('\n✅ Accuracy summary saved to results/accuracy_summary.csv')
## 💾 Step 21: Save & Export Model
# Save in H5 format
model.save(CONFIG['model_save_path'])
print(f'✅ Model saved: {CONFIG["model_save_path"]}')

# Save in TensorFlow SavedModel format (for deployment)
model.save('./saved_model/multicrop_savedmodel')
print('✅ SavedModel format saved: ./saved_model/multicrop_savedmodel/')

# Convert to TFLite (for mobile/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('./saved_model/multicrop_model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f'✅ TFLite model saved: ./saved_model/multicrop_model.tflite')
print(f'   TFLite size: {len(tflite_model)/1024/1024:.2f} MB')
## ✅ Step 22: Project Summary

### What was built:
| Component | Details |
|---|---|
| **Model** | EfficientNetB3 + Custom Head (TensorFlow/Keras) |
| **Input** | 224×224 RGB leaf images |
| **Crops** | Rice, Wheat, Maize, Tomato, Potato, Cotton, Banana, Sugarcane, Ash Gourd, Cherry |
| **Output** | Crop + Stress type + Confidence + Top-3 predictions |
| **Recommendations** | Traditional remedies + Organic fertilizer + Chemical fertilizer |
| **Saved formats** | `.h5`, `SavedModel`, `.tflite` |

### Output files in `results/`:
- `training_history.png` — accuracy/loss curves
- `confusion_matrix.png` — per-class confusion matrix
- `f1_scores.png` — precision/recall/F1 bar chart
- `classification_report.json` — full per-class metrics
- `accuracy_summary.csv` — summary table
- `class_indices.json` — label mapping

### Next steps:
- Deploy API with FastAPI (`app.py`)
- Add more crop images for underperforming classes
- Try EfficientNetB5 or Vision Transformer for higher accuracy
- Integrate severity scoring (mild / moderate / severe)
