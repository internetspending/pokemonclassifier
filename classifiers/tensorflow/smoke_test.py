import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

NUM_CLASSES = 18
IMG_SIZE = (224, 224)


def build_model():
    """Build EfficientNetB0 model"""
    print("[INFO] Building model...")
    
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # Load pretrained base
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None
    )
    
    base_model.trainable = False
    
    # Custom head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def main():
    print("="*60)
    print("TENSORFLOW SMOKE TEST - POKEMON CLASSIFIER")
    print("="*60)
    
    # Check TensorFlow version
    print(f"\n[INFO] TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("[INFO] Running on CPU")
    
    # Build model
    model = build_model()
    
    print(f"\n[INFO] Model built successfully")
    print(f"[INFO] Total parameters: {model.count_params():,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[INFO] Model compiled successfully")
    
    # Create dummy input
    print(f"\n[INFO] Testing forward pass with dummy input...")
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
    
    # Forward pass
    output = model.predict(dummy_input, verbose=0)
    
    print(f"[INFO] Input shape: {dummy_input.shape}")
    print(f"[INFO] Output shape: {output.shape}")
    print(f"[INFO] Expected output shape: (1, {NUM_CLASSES})")
    
    # Verify output
    assert output.shape == (1, NUM_CLASSES), "Output shape mismatch!"
    assert np.allclose(output.sum(), 1.0, atol=1e-5), "Output probabilities should sum to 1!"
    
    # Show top predictions
    top_indices = np.argsort(output[0])[::-1][:3]
    print("\n[INFO] Top 3 predictions (random output):")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Class {idx}: {output[0][idx]:.4f}")


if __name__ == '__main__':
    main()