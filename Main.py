# Library imports
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import torch
import tiktoken as tik

# My class imports
from TokeniserClass import Tokeniser
import GPTDataLoaderClass

# main body
train = pd.read_csv('./Data/CustomerServiceDataSet.csv')
tokeniserRef = Tokeniser(train)

texter = ""

for i in range (len(train['instruction'])-1):
    texter += (train['instruction'][i] + " ")

#print(texter)

# testing my tokeniser class
#test = tokeniserRef.TokeniseText(texter)
#print(test)
#test2 = tokeniserRef.DetokeniseArray(test)
#print(test2)

# testing the tiktokeniser
#OtherTokeniser = tik.get_encoding("gpt2")
#coded = OtherTokeniser.encode(texter)
#print(coded)
#decoded = OtherTokeniser.decode(coded)
#print(decoded)

# testing the data loader for better tokenising
#dataloader = GPTDataLoaderClass.CreateDataLoader(texter, batchSize=8, maxLength=4, stride=4, shuffleData=False)
#dataIter = iter(dataloader)
#inputs, targets = next(dataIter)
#print("Inputs:\n", inputs)
#print("\nTargets:\n", targets)

#testing making an embedding layer
inputIds = torch.tensor([2,3,5,1])
vocabSize = 6
outputDimensions = 3

torch.manual_seed(123)
embeddingLayer = torch.nn.Embedding(vocabSize, outputDimensions)
print(embeddingLayer(inputIds))