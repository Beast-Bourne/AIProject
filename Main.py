import numpy as np
import pandas as pd
import re
import tensorflow as tf
import torch as tch

from TokeniserClass import Tokeniser

train = pd.read_csv('./Data/CustomerServiceDataSet.csv')
vocab = []

# read all of the instructions that customers provided
# splits them and adds each word to an array
for i in range(len(train)):
    text = train['instruction'][i]
    spliter = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    spliter = [item for item in spliter if item.strip()]
    vocab += spliter

# sorts the vocab array alphabetically then removes all duplicate entries
vocab = sorted(set(vocab))
vocab = {token:integer for integer, token in enumerate(vocab)}


# pass a string in to tokeniser['STRING'] to get a token (a unique integer identifier)
# pass an integer into detokeniser to get its corrosponding string
# This only works for words and tokens in the 'vocab' dataset
tokeniser = vocab
detokeniser = {i:s for s,i in vocab.items()}

tokeniserRef = Tokeniser(vocab)

texter = train['instruction'][4]
print(texter)

test = tokeniserRef.TokeniseText(texter)
print(test)

test2 = tokeniserRef.DetokeniseArray(test)
print(test2)

