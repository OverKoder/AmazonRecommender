from config.config import RAW_DATA_PATH, CATEGORY2IDX
from data.dataset import AmazonDataset
from data.preprocessing import preprocess, get_data
from models.tinymodel import TinyModel

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import mlflow

def main():

    # Load data
    data, targets = preprocess(get_data(RAW_DATA_PATH))

    # Split data
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(data, targets, train_size = 0.7, random_state = 42)
    val_data, _, val_labels, _ = train_test_split(val_test_data, val_test_labels, test_size = 0.5, random_state = 42)

    # Datasets and loaders
    train_dataset = AmazonDataset(train_data, train_labels)
    val_dataset = AmazonDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True)

    # Model and wrapper
    model = AutoModelForSequenceClassification.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification", num_labels = len(CATEGORY2IDX), ignore_mismatched_sizes=True)
    model = TinyModel(model, len(CATEGORY2IDX))

    # Callback
    early_stopping = EarlyStopping('val_acc', mode = "max", patience = 1, min_delta = 0.025)


    # Trainer
    trainer = Trainer(accelerator='gpu', devices = 1, max_epochs=10, enable_progress_bar=True, callbacks = [early_stopping])

    # Train
    mlflow.set_experiment("FeverChallenge")
    with mlflow.start_run() as run:
        mlflow.pytorch.autolog()

        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()