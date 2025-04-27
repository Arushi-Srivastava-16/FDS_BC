# breast_cancer_detection/configs/config.py

class Config:
    # Dataset paths
    IMAGE_DIR = '/path/to/vindr/images'
    ANNOTATIONS_PATH = '/path/to/vindr/annotations.json'

    # Training hyperparameters
    BATCH_SIZE = 8
    IMAGE_SIZE = 512
    NUM_WORKERS = 4
    NUM_QUERIES = 10  # Number of proposals per image

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Model settings
    BACKBONE = 'resnet50'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Contrastive Learning
    CONTRASTIVE_PROJECTION_DIM = 256
    TEMPERATURE = 0.5

    # Training
    NUM_EPOCHS = 50
    SAVE_MODEL_PATH = './checkpoints/'

