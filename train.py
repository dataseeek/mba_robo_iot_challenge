"""
Training CLI for person detection using Transfer Learning on MobileNet.

Faithfully replicates the model-train.ipynb pipeline:
  1. Load dataset from directory structure (NENHUM/, PESSOA/)
  2. Build MobileNet-based model cut at conv_pw_10_relu
  3. Phase 1: Train with frozen base (40 epochs)
  4. Phase 2: Fine-tune entire model (20 epochs)
  5. Export SavedModel + TFLite INT8 quantized model

Usage:
  python train.py --dataset ./dataset --output ./output
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    CLASS_DIRS, CLASSES, NUM_CLASSES, IMAGE_DIM, MOBILENET_ALPHA, CUT_LAYER,
    FROZEN_EPOCHS, FROZEN_LR, FINETUNE_EPOCHS, FINETUNE_LR,
    BATCH_SIZE, DROPOUT_RATE, VALIDATION_SPLIT, RANDOM_STATE,
    ROTATION_RANGE, ZOOM_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP,
)


# Data augmentation as a Keras Sequential model (replaces ImageDataGenerator)
data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(ROTATION_RANGE / 360),
    keras.layers.RandomZoom(ZOOM_RANGE),
    keras.layers.RandomTranslation(HEIGHT_SHIFT_RANGE, WIDTH_SHIFT_RANGE),
    keras.layers.RandomFlip("horizontal" if HORIZONTAL_FLIP else "none"),
])


def build_model() -> keras.Model:
    """Build MobileNet transfer learning model (notebook cell 14 + 16)."""
    base_model = keras.applications.MobileNet(
        weights="imagenet",
        input_shape=(IMAGE_DIM, IMAGE_DIM, 3),
        alpha=MOBILENET_ALPHA,
        include_top=False,
    )
    base_model.trainable = False

    last_layer = base_model.get_layer(CUT_LAYER)

    x = keras.layers.Reshape((-1, last_layer.output.shape[3]))(last_layer.output)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(base_model.input, x)
    return model


def create_datasets(dataset_path: str, batch_size: int):
    """Create train/validation tf.data.Datasets from directory structure."""
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        image_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_STATE,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_DIRS,
        image_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_STATE,
        shuffle=False,
    )

    # Count samples
    train_count = train_ds.cardinality().numpy() * batch_size
    val_count = val_ds.cardinality().numpy() * batch_size

    # Rescale [0,255] → [0,1] (matching notebook cell 6: data / 255.0)
    rescale = keras.layers.Rescaling(1.0 / 255)

    # Training: rescale + augment + prefetch
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(rescale(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    # Validation: rescale + prefetch only
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_count, val_count


def train_frozen(model: keras.Model, train_ds, val_ds, epochs: int):
    """Phase 1: Train with frozen base (notebook cell 18)."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Frozen base training ({epochs} epochs)")
    print(f"{'='*60}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FROZEN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
    )
    return history


def train_finetune(model: keras.Model, train_ds, val_ds, epochs: int):
    """Phase 2: Fine-tune entire model (notebook cell 25)."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tuning ({epochs} epochs)")
    print(f"{'='*60}\n")

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
    )
    return history


def export_savedmodel(model: keras.Model, output_dir: str):
    """Export as SavedModel (notebook cell 33)."""
    savedmodel_path = os.path.join(output_dir, "person-detect-model.keras")
    print(f"\nSaving model to {savedmodel_path} ...")
    model.save(savedmodel_path)
    print("Model saved.")
    return savedmodel_path


def export_tflite_int8(model: keras.Model, train_ds, output_dir: str):
    """Export INT8 quantized TFLite model (notebook cells 35-37)."""
    tflite_path = os.path.join(output_dir, "person-detect-model-int8.tflite")
    print(f"\nQuantizing INT8 TFLite model to {tflite_path} ...")

    # Collect calibration images from training dataset (already rescaled to [0,1])
    calibration_images = []
    for batch_images, _ in train_ds.take(2):
        for img in batch_images:
            calibration_images.append(img.numpy())
            if len(calibration_images) >= BATCH_SIZE:
                break
        if len(calibration_images) >= BATCH_SIZE:
            break
    calibration_data = np.array(calibration_images, dtype="float32")

    def representative_dataset_gen():
        for sample in tf.data.Dataset.from_tensor_slices(calibration_data).batch(1).take(BATCH_SIZE):
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"INT8 TFLite model saved ({len(tflite_model)} bytes).")
    return tflite_path


def export_tflite_float16(model: keras.Model, output_dir: str):
    """Export float16 quantized TFLite model — better accuracy, ~2x size vs INT8."""
    tflite_path = os.path.join(output_dir, "person-detect-model.tflite")
    print(f"\nQuantizing float16 TFLite model to {tflite_path} ...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Float16 TFLite model saved ({len(tflite_model)} bytes).")
    return tflite_path


def main():
    parser = argparse.ArgumentParser(
        description="Train person detection model (MobileNet transfer learning)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset directory containing NENHUM/ and PESSOA/ subdirs",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Output directory for models (default: ./output)",
    )
    parser.add_argument("--frozen-epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-finetune", action="store_true", help="Skip fine-tuning phase")
    parser.add_argument("--no-quantize", action="store_true", help="Skip TFLite export")
    args = parser.parse_args()

    # Validate dataset
    for cls in CLASS_DIRS:
        cls_path = os.path.join(args.dataset, cls)
        if not os.path.isdir(cls_path):
            print(f"ERROR: Missing class directory: {cls_path}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Build model
    print("Building model...")
    model = build_model()
    model.summary()

    # Create datasets
    train_ds, val_ds, train_count, val_count = create_datasets(args.dataset, args.batch_size)
    print(f"\nTraining samples: ~{train_count}")
    print(f"Validation samples: ~{val_count}")

    # Phase 1: Frozen training
    train_frozen(model, train_ds, val_ds, args.frozen_epochs)

    # Phase 2: Fine-tuning
    if not args.no_finetune:
        train_finetune(model, train_ds, val_ds, args.finetune_epochs)

    # Evaluate
    print("\nFinal evaluation:")
    loss, acc = model.evaluate(val_ds)
    print(f"Validation loss: {loss:.4f}, accuracy: {acc:.4f}")

    # Export model
    export_savedmodel(model, args.output)

    # Export TFLite (float16 = default, int8 = optional for ultra-constrained devices)
    if not args.no_quantize:
        export_tflite_float16(model, args.output)
        export_tflite_int8(model, train_ds, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
