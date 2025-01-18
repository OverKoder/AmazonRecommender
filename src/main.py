from data.preprocessing import get_data, preprocess
from data.dataset import AmazonDataset

def main():

    print("Debugging info")
    data, targets = preprocess(get_data('data.gz'))
    print(targets.shape)
    dataset = AmazonDataset(data, targets)
    print(dataset.__len__)
    inputs, targets = dataset.__getitem__(0)
    print(inputs)
    print(targets)


    
if __name__ == "__main__":
    main()