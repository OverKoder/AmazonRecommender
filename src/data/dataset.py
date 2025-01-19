from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch import from_numpy

class AmazonDataset(Dataset):

    def __init__(self, data, targets):
        super().__init__()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification")

        # Data and targets
        self.data = data
        self.targets = from_numpy(targets)

        return
        
    
    def __getitem__(self, index):

        # Tokenize reviews
        review = self.tokenizer(self.data[index], return_tensors="pt", max_length = 512, padding = "max_length", truncation = True)
        target = self.targets[index]

        return review, target

    def __len__(self):
        return self.targets.shape[0]