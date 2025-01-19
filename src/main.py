from data.preprocessing import get_data, preprocess
from data.dataset import AmazonDataset
from transformers import BertTokenizerFast
def main():

    print("Debugging info")
    data, targets = preprocess(get_data('data.gz'))
    print(targets.shape)
    
    max_length = max([len(elem) for elem in data])
    max_elem = [elem for elem in data if len(elem) == max_length][0]
    seq_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    print(seq_tokenizer(max_elem, return_tensors="pt", max_length = 512, padding = "max_length")['input_ids'])

    
if __name__ == "__main__":
    main()