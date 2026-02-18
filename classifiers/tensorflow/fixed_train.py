import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuration
DATA_DIR = 'data'  
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 18
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 25
LEARNING_RATE = 1e-3

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_csv():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / DATA_DIR / "pokemon.csv"

    df = pd.read_csv(data_path)
    
    # Create label mappings
    label_names = sorted(df['Type1'].unique())
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    
    print(f"  Classes: {label_names}")
    
    return df, label_names, label_to_idx


def create_splits(df):
    """Create train/val/test splits"""
    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Type1']
    )
    
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=42, stratify=train_val['Type1']
    )
    
    return train, val, test


def load_and_preprocess_image(img_path, label):
    """Load and preprocess image - handles RGBA to RGB"""
    # Read image file
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=4)  # RGBA
    
    # RGBA → RGB with white background
    rgb = img[:, :, :3]
    alpha = tf.cast(img[:, :, 3:], tf.float32) / 255.0
    white_bg = tf.ones_like(rgb, dtype=tf.float32) * 255.0
    rgb_float = tf.cast(rgb, tf.float32)
    img_rgb = rgb_float * alpha + white_bg * (1 - alpha)
    img_rgb = tf.cast(img_rgb, tf.uint8)
    
    # Resize and normalize
    img_rgb = tf.image.resize(img_rgb, [IMG_SIZE, IMG_SIZE])
    img_rgb = tf.cast(img_rgb, tf.float32) / 255.0
    
    return img_rgb, label


def augment_image(img, label):
    """Data augmentation for training"""
    # Random horizontal flip
    img = tf.image.random_flip_left_right(img)
    
    # Random brightness
    img = tf.image.random_brightness(img, max_delta=0.1)
    
    # Random contrast
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    
    # Clip to valid range
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, label


def create_dataset(df, label_to_idx, is_training=True):
    """Create TensorFlow dataset"""
    # Get image paths and labels
    project_root = Path(__file__).resolve().parent.parent.parent

    img_paths = [
        os.path.join(project_root / DATA_DIR, 'images', f"{name}.png") 
        for name in df['Name']
    ]
    labels = [label_to_idx[t] for t in df['Type1']]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()  # Repeat for multiple epochs
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(img_paths)


def build_model():
    """Build EfficientNetB0 model with proper initialization"""
    # Input layer
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocess for EfficientNet (normalize to [-1, 1])
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Load pretrained base
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling=None
    )
    base.trainable = False  # Freeze initially
    
    # Add classification head
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)  # Additional dense layer
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, base


def train_phase1(model, train_dataset, val_dataset, train_size, val_size):
    """Phase 1: Train classification head only"""
    print("\n[PHASE 1] Training classification head (base frozen)...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS_PHASE1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
    )
    
    return history


def train_phase2(model, base_model, train_dataset, val_dataset, train_size, val_size):
    """Phase 2: Fine-tune upper layers"""
    print("\n[PHASE 2] Fine-tuning upper layers...")
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze early layers (only fine-tune last 50 layers)
    # FIXED: Properly access the actual EfficientNet model
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - 50)
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"  Unfrozen {trainable_count}/{total_layers} layers in base model")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS_PHASE2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7
            )
        ]
    )
    
    return history


def evaluate_model(model, test_dataset, test_labels, label_names):
    """Evaluate model on test set"""
    print("\n[EVALUATION] Testing model...")
    
    # Get predictions
    predictions = model.predict(test_dataset)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(pred_classes == test_labels)
    
    # Top-3 accuracy
    top3_correct = 0
    for i, true_label in enumerate(test_labels):
        top3_preds = np.argsort(predictions[i])[-3:]
        if true_label in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(test_labels)
    
    # Per-class accuracy
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    
    # Show some predictions
    print("\n  Sample predictions:")
    for i in range(min(5, len(test_labels))):
        true_label = label_names[test_labels[i]]
        pred_label = label_names[pred_classes[i]]
        confidence = predictions[i][pred_classes[i]]
        print(f"    True: {true_label:>10} | Predicted: {pred_label:>10} ({confidence:.3f})")
    
    return accuracy, top3_acc


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("POKEMON CLASSIFIER - TENSORFLOW (FIXED)")
    print("=" * 60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n[INFO] GPU Available: {len(gpus) > 0}")
    
    # Load data
    print("\n[STEP 1] Loading CSV...")
    df, label_names, label_to_idx = load_csv()
    print(f"  Loaded {len(df)} Pokemon, {len(label_names)} types")
    
    # Create splits
    print("\n[STEP 2] Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create datasets
    print("\n[STEP 3] Creating TensorFlow datasets...")
    train_dataset, train_size = create_dataset(train_df, label_to_idx, is_training=True)
    val_dataset, val_size = create_dataset(val_df, label_to_idx, is_training=False)
    test_dataset, test_size = create_dataset(test_df, label_to_idx, is_training=False)
    
    # Build model
    print("\n[STEP 4] Building EfficientNetB0 model...")
    model, base_model = build_model()
    print(f"  Total parameters: {model.count_params():,}")
    
    # Train Phase 1
    print("\n[STEP 5] Training model...")
    history1 = train_phase1(model, train_dataset, val_dataset, train_size, val_size)
    
    # Train Phase 2
    history2 = train_phase2(model, base_model, train_dataset, val_dataset, train_size, val_size)
    
    # Evaluate
    test_labels = np.array([label_to_idx[t] for t in test_df['Type1']])
    accuracy, top3_acc = evaluate_model(model, test_dataset, test_labels, label_names)
    
    # Save model
    model.save('pokemon_model_fixed.keras')
    print(f"\n[SAVED] Model saved to 'pokemon_model_fixed.keras'")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Final Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()