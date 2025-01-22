import gzip
import json
from pathlib import Path
import os
import sys

from config.config import CATEGORY2IDX

import pandas as pd
import numpy as np
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

def parse(path):
    """
    Generator which yields reviews in the data.
    """
    # Check if path exists
    assert (Path.cwd() / path).exists(), f"{Path.cwd() / path} not found, please check the path."

    # Load data
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def get_data(path):
    """
    Loads data and return a pandas DataFrame.
    """
    data = {}
    for i, d in enumerate(parse(path)):
        data[i] = d
        i += 1

    return pd.DataFrame.from_dict(data, orient='index')


def preprocess(data: pd.DataFrame):
    """
    Preprocessing pipeline
    """

    # Current design choice: Ignore also_buy, also_view, asin, images and price columns
    # 52% of brands are included in the title
    data.drop(columns=['also_buy', 'also_view', 'asin', 'category', 'image', 'price'], inplace = True)

    # Split into data and targets
    targets = data['main_cat'].apply(lambda x: CATEGORY2IDX[x])
    data.drop(columns = ['main_cat'], inplace = True)

    lengths = []
    for i, row in enumerate(data.iterrows()):

        row = row[1].values
        sequence = []
        for elem in row:

            if isinstance(elem, str):
                sequence.append(elem.replace('.',''))

            else:
                for subelem in elem:
                    sequence.append(subelem.replace('.',''))

        # Join all sequence parts
        sequence = '. '.join(sequence)
        data.loc[i, 'sequence'] = sequence
        lengths.append(len(sequence))

    data['lengths'] = lengths

    # Remove rows with length outside of quantile
    remove_index = data[data['lengths'] > data['lengths'].quantile(0.99)].index
    data.drop(remove_index, inplace=True)
    targets.drop(remove_index, inplace = True)

    # Get remaining sequences
    data = data['sequence'].values

    return data, targets.to_numpy()