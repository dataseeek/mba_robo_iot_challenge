"""Shared constants for person detection training and inference."""

# Classes — binary classification (index 0 = no person, index 1 = person)
CLASS_DIRS = ["NENHUM", "PESSOA"]       # dataset directory names (used by train.py)
CLASSES = ["NENHUMA", "PESSOA"]         # display names (used by api.py)
NUM_CLASSES = len(CLASSES)

# MobileNet architecture (matching notebook exactly)
IMAGE_DIM = 96
MOBILENET_ALPHA = 0.25
CUT_LAYER = "conv_pw_10_relu"

# Training defaults — Phase 1 (frozen base)
FROZEN_EPOCHS = 40
FROZEN_LR = 0.0005

# Training defaults — Phase 2 (fine-tuning)
FINETUNE_EPOCHS = 20
FINETUNE_LR = 0.00001

# Common training params
BATCH_SIZE = 100
DROPOUT_RATE = 0.1
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 0

# Data augmentation
ROTATION_RANGE = 10
ZOOM_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True

# Inference defaults
CONFIDENCE_THRESHOLD = 0.6
CAMERA_FPS = 5
