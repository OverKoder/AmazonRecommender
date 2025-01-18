from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from torch import from_numpy

class AmazonDataset(Dataset):

    def __init__(self, data, targets):
        super().__init__()

        # Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

        # Data and targets
        self.data = self.tokenizer(data, return_tensors="pt", padding=True)
        self.targets = from_numpy(targets)

        return
        
    
    def __getitem__(self, index):

        # Reviews are already tokenized
        review = self.data[index]
        target = self.targets[index]

        return review, target

    def __len__(self):
        return self.targets.shape[0]