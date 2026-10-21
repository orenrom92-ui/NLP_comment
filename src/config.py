from pathlib import Path
import torch

# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Модель
MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

# Токенизация
MAX_LENGTH = 256

# Классы
NUM_CLASSES = 3
LABEL_NAMES = ["negative", "neutral", "positive"]

# Устройство
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
NUM_EPOCHS = 5
RANDOM_SEED = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05