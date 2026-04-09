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
out = model(batch)
print(out.shape)



# ######################################################################################################################## token weighting testing
# # combinging the position and token embeddings gives all the information necessary for the GPT model to use

# # this is the word being queried. The model will use an attention mechanism to determine how much attention to pay to each of the other words in the input string when processing this word
# queryWord = inputEmbeddings[0][0]

# # attention scores are determined by taking the dot product of the query word with each of the other words in the input string. (the embeddings are dotted together to determine how similar they are)
# attn_scores = torch.empty(inputEmbeddings[0].shape[0])
# for i, x_i in enumerate(inputEmbeddings[0]):
#     attn_scores[i] = torch.dot(x_i, queryWord)

# # the scores are normalised to give attention weightings that sum to 1
# attn_weights = torch.softmax(attn_scores, dim=0)

# # the final context vector is a weighted sum of the word's embeddings, where the weights are determined by the attention mechanism. This context vector is what the model uses to make predictions about the next word in the sequence
# contextVector = torch.empty(inputEmbeddings[0][0].shape[0])
# for i, x_i in enumerate(inputEmbeddings[0]):
#     contextVector += attn_weights[i] * x_i


# ######################################################################################################################## fast token weight calculations
# # fast matrix multiplication to get the attention weights (does everything in the above section at once for each word)
# temp = inputEmbeddings[0] @ inputEmbeddings[0].T
# tempW = torch.softmax(temp, dim=1)

# # context vectors for every word
# tempC = tempW @ inputEmbeddings[0]


# ######################################################################################################################## training weights

# queryWordChoice = inputEmbeddings[0][0]
# wordShape = inputEmbeddings[0][0].shape[0]
# weightShape = 2

# torch.manual_seed(123)
# queryWeight = torch.nn.Linear(wordShape, weightShape, bias=False)
# keyWeight = torch.nn.Linear(wordShape, weightShape, bias=False)
# valueWeight = torch.nn.Linear(wordShape, weightShape, bias=False)

# queryTest = queryWeight(queryWordChoice) # this weight is based on the query word

# keys = keyWeight(inputEmbeddings[0]) # these weights are unique to each other word
# values = valueWeight(inputEmbeddings[0]) # these values are unique to each other word

# queryAttentionScores = queryTest @ keys.T # the attention scores are based on the dot product of the query and key weights

# d_k = keys.shape[1] # this is the dimension of the key vectors, used for scaling the attention scores to prevent them from getting too large
# queryAttentWeights = torch.softmax(queryAttentionScores / d_k**0.5, dim=-1) # the attention weights are the normalised attention scores

# queryContextVector = queryAttentWeights @ values # the context vector is a weighted sum of the value vectors, where the weights are determined by the attention mechanism
# print(queryContextVector)

# ######################################################################################################################## generalise for all input words

# print(inputEmbeddings)

# contLength = inputEmbeddings.shape[1]
# thing1 = SelfAttention(wordShape, weightShape, contLength)
# #print(thing1(inputEmbeddings))


# torch.manual_seed(123)
# contextLength = inputEmbeddings.shape[1]
# d_in = inputEmbeddings.shape[2]
# d_out = 4
# mha = MultiHeadAttentionWrapper(d_in, d_out, contextLength, 2)
# print(mha(inputEmbeddings))
# print(mha(inputEmbeddings).shape)