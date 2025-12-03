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

texter = ""

for i in range (len(train['instruction'])-1):
    texter += (train['instruction'][i] + " ")

# finding the shortest instruction in the dataset
# shortest = 0
# index = 0
# for i in range (len(train['instruction'])-1):
#     #if (train['intent'][i] != "cancel_order"): continue

#     text = train['instruction'][i]
#     spliter = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#     spliter = [item for item in spliter if item.strip()]
#     if (len(spliter) > shortest): 
#         shortest = len(spliter)
#         index = i

# print(index)
# print(train['instruction'][index])
# print(train['response'][index])

# testing my tokeniser class
#tokeniserRef = Tokeniser(train)
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

######################################################################################################################## ctrl + / to mass comment out
# testing the data loader for better tokenising
dataloader = GPTDataLoaderClass.CreateDataLoader(texter, batchSize=8, maxLength=4, stride=4, shuffleData=False)
dataIter = iter(dataloader)
inputs, targets = next(dataIter)
#print("Inputs:\n", inputs)
#print("\nTargets:\n", targets)

#testing making an embedding layer
vocabSize = tik.get_encoding("gpt2").n_vocab
outputDimensions = 256
torch.manual_seed(123)
embeddingLayer = torch.nn.Embedding(vocabSize, outputDimensions)

# token embeddings give information about the token itself
tokenEmbeddings = embeddingLayer(inputs)

contextLength = 4
positionEmbeddingLayer = torch.nn.Embedding(contextLength, outputDimensions)

#position embedding gives information about the position of the token in its string
positionEmbeddings = positionEmbeddingLayer(torch.arange(contextLength))

# combinging the position and token embeddings gives all the information necessary for the GPT model to use
inputEmbeddings = tokenEmbeddings + positionEmbeddings
print(inputEmbeddings.shape)