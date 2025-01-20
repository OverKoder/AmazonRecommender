from config.config import RAW_DATA_PATH, WEIGHTS_PATH, CATEGORY2IDX
from data.dataset import AmazonDataset
from data.preprocessing import preprocess, get_data
from models.tinymodel import TinyModel

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import AutoModelForSequenceClassification

# Load data
data, targets = preprocess(get_data(RAW_DATA_PATH))

# Split data
train_data, val_test_data, train_labels, val_test_labels = train_test_split(data, targets, train_size = 0.7, random_state = 42)
val_data, test_data, val_labels, test_labels = train_test_split(val_test_data, val_test_labels, test_size = 0.5, random_state = 42)

# Datasets and loaders
test_dataset = AmazonDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True)

# Model (test accuracy = 0.8565)
model = AutoModelForSequenceClassification.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification", num_labels = len(CATEGORY2IDX), ignore_mismatched_sizes=True)
model = TinyModel(model = model, num_classes = len(CATEGORY2IDX))

# Trainer
trainer = Trainer(accelerator='gpu', devices = 1, max_epochs=1, enable_progress_bar=True)
metrics = trainer.test(model, test_loader, WEIGHTS_PATH)
print(metrics)