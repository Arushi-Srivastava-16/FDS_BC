import torch
class Config:
    IMAGE_DIR = 'vindrpng/images_png/'  # Change to your real image path
    ANNOTATIONS_PATH = 'filtered_has_finding.csv'
    BATCH_SIZE = 8
    IMAGE_SIZE = 512
    NUM_WORKERS = 2
    NUM_QUERIES = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BACKBONE = 'resnet50'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONTRASTIVE_PROJECTION_DIM = 256
    TEMPERATURE = 0.5
    NUM_EPOCHS = 50
    SAVE_MODEL_PATH = './checkpoints/'
