import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuration
DATA_DIR = 'data'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 18
EPOCHS = 30
LEARNING_RATE = 1e-3


def load_csv():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / DATA_DIR / "pokemon.csv"
    """Load CSV and create label mappings (same as PyTorch)"""
    df = pd.read_csv(data_path)
    
    # Create label mappings
    label_names = sorted(df['Type1'].unique())
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    
    return df, label_names, label_to_idx


def create_splits(df):
    """Create train/val/test splits"""
    # Train+Val vs Test
    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Type1']
    )
    
    # Train vs Val
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=42, stratify=train_val['Type1']
    )
    
    return train, val, test


def load_and_preprocess_image(img_path, label):
    """
    Load and preprocess image (similar to utils/preprocessing.py)
    Handles RGBA → RGB conversion like PyTorch version
    """
    # Read image file
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=4)  # RGBA
    
    # RGBA → RGB with white background (same logic as PyTorch)
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


def create_dataset(df, label_to_idx, is_training=True):
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
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_model():
    # Input layer
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Load pretrained base (like PyTorch's efficientnet_b0(weights='DEFAULT'))
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    base.trainable = False  # Freeze initially
    
    # Add classification head (like PyTorch's model.classifier[1] = nn.Linear(...))
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def train_model(model, train_dataset, val_dataset):

    print("\n[PHASE 1] Training classification head (base frozen)...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n[PHASE 2] Fine-tuning upper layers...")
    
    # Unfreeze top layers 
    base = model.layers[1]
    base.trainable = True
    for layer in base._layers[:-50]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        verbose=1
    )
    
    return model


def evaluate_model(model, test_dataset, test_labels):
    print("\n[EVALUATION] Testing model...")
    
    predictions = model.predict(test_dataset)
    pred_classes = np.argmax(predictions, axis=1)
    
    accuracy = np.mean(pred_classes == test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy


def main():
    """Main pipeline"""
    print("=" * 60)
    print("POKEMON CLASSIFIER - TENSORFLOW (SIMPLIFIED)")
    print("=" * 60)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n[INFO] GPU Available: {len(gpus) > 0}")
    
    # Step 1: Load data (like your PyTorch code)
    print("\n[STEP 1] Loading CSV...")
    df, label_names, label_to_idx = load_csv()
    print(f"  Loaded {len(df)} Pokemon, {len(label_names)} types")
    
    # Step 2: Create splits
    print("\n[STEP 2] Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Step 3: Create datasets
    print("\n[STEP 3] Creating TensorFlow datasets...")
    train_dataset = create_dataset(train_df, label_to_idx, is_training=True)
    val_dataset = create_dataset(val_df, label_to_idx, is_training=False)
    test_dataset = create_dataset(test_df, label_to_idx, is_training=False)
    
    # Step 4: Build model
    print("\n[STEP 4] Building EfficientNetB0 model...")
    model = build_model()
    print(f"  Total parameters: {model.count_params():,}")
    
    # Step 5: Train
    print("\n[STEP 5] Training model...")
    model = train_model(model, train_dataset, val_dataset)
    
    # Step 6: Evaluate
    test_labels = np.array([label_to_idx[t] for t in test_df['Type1']])
    accuracy = evaluate_model(model, test_dataset, test_labels)
    
    # Step 7: Save
    model.save('pokemon_model.keras')
    print(f"\n[SAVED] Model saved to 'pokemon_model.keras'")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE - Final Accuracy: {accuracy:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()