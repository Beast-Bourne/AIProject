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
from SelfAttentionClass import SelfAttention
from SelfAttentionClass import MultiHeadAttentionWrapper
import GPTModelClass as GPT

# main body
train = pd.read_csv('./Data/CustomerServiceDataSet.csv')
texter = ""

for i in range (len(train['instruction'])-1):
    texter += (train['instruction'][i] + " ")

# ctrl + / to mass comment out
######################################################################################################################## token embedding testing
slidingWindowLength = 4

# testing the data loader for better tokenising
dataloader = GPTDataLoaderClass.CreateDataLoader(texter, batchSize=2, maxLength=slidingWindowLength, stride=3, shuffleData=False)
dataIter = iter(dataloader)
inputs, targets = next(dataIter)

# testing making an embedding layer
vocabSize = tik.get_encoding("gpt2").n_vocab
outputDimensions = 3
torch.manual_seed(123)
embeddingLayer = torch.nn.Embedding(vocabSize, outputDimensions)

# token embeddings give information about the token itself
tokenEmbeddings = embeddingLayer(inputs)

positionEmbeddingLayer = torch.nn.Embedding(slidingWindowLength, outputDimensions)

#position embedding gives information about the position of the token in its string
positionEmbeddings = positionEmbeddingLayer(torch.arange(slidingWindowLength))
inputEmbeddings = tokenEmbeddings + positionEmbeddings


######################################################################################################################## Testing stuff

tokeniser = tik.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokeniser.encode(txt1)))
batch.append(torch.tensor(tokeniser.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPT.GPTModel(GPT.GPT_CONFIG)
# out = model(batch)
# print(out.shape)

def generateTextTest(model, inputText, maxNewTokens, contextSize):
    for _ in range(maxNewTokens):
        inputCondition = inputText[:, -contextSize:]

        with torch.no_grad():
            logits = model(inputCondition)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        nextToken = torch.argmax(probs, dim=-1, keepdim=True)

        inputText = torch.cat((inputText, nextToken), dim=1)

    return inputText

temp = "Hello, I am"
encoded = tokeniser.encode(temp)
encodedTensor = torch.tensor(encoded).unsqueeze(0)

out = generateTextTest(model, encodedTensor, 6, 1024)
outText = tokeniser.decode(out.squeeze(0).tolist())
print(outText)