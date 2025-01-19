from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class TinyModel(pl.LightningModule):
    """
    Wrapper class for LongFormer model.
    """
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.metric = MulticlassAccuracy(num_classes = num_classes, average = 'macro')

        return

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.model(input_ids = inputs['input_ids'].squeeze(1), attention_mask = inputs['attention_mask'].squeeze(1), labels = labels)

        # Compute loss and log it
        self.log("train_loss", outputs.loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        train_acc = self.metric(outputs.logits, labels)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.model(input_ids = inputs['input_ids'].squeeze(1), attention_mask = inputs['attention_mask'].squeeze(1), labels = labels)

        # Compute loss and log it
        self.log("val_loss", outputs.loss, prog_bar=True, on_epoch=True)

        # Log F1-Score
        val_acc = self.metric(outputs.logits, labels)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True)

        return outputs.loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)