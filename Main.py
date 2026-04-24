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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# # ctrl + / to mass comment out
# ######################################################################################################################## token embedding testing
# slidingWindowLength = 4

# # testing the data loader for better tokenising
# dataloader = GPTDataLoaderClass.CreateDataLoader(texter, batchSize=2, maxLength=slidingWindowLength, stride=3, shuffleData=False)
# dataIter = iter(dataloader)
# inputs, targets = next(dataIter)

# # testing making an embedding layer
# vocabSize = tik.get_encoding("gpt2").n_vocab
# outputDimensions = 3
# torch.manual_seed(123)
# embeddingLayer = torch.nn.Embedding(vocabSize, outputDimensions)

# # token embeddings give information about the token itself
# tokenEmbeddings = embeddingLayer(inputs)

# positionEmbeddingLayer = torch.nn.Embedding(slidingWindowLength, outputDimensions)

# #position embedding gives information about the position of the token in its string
# positionEmbeddings = positionEmbeddingLayer(torch.arange(slidingWindowLength))
# inputEmbeddings = tokenEmbeddings + positionEmbeddings


######################################################################################################################## Testing stuff

# txt1 = "Every effort moves you"
# tokeniser = tik.get_encoding("gpt2")
# batch = []

# torch.manual_seed(123)
# model = GPT.GPTModel(GPT.GPT_CONFIG)
# model.eval()

def TextToTokenIds(text, tokeniser):
    encoded = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
    encodedTensor = torch.tensor(encoded).unsqueeze(0)
    return encodedTensor


#batch = TextToTokenIds(txt1, tokeniser)

def TokenIdsToText(tokenIds, tokeniser):
    flat = tokenIds.squeeze(0).tolist()
    return tokeniser.decode(flat)


def GenerateTextTest(model, inputText, maxNewTokens, contextSize):
    for _ in range(maxNewTokens):
        inputCondition = inputText[:, -contextSize:]

        with torch.no_grad():
            logits = model(inputCondition)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        nextToken = torch.argmax(probs, dim=-1, keepdim=True)

        inputText = torch.cat((inputText, nextToken), dim=1)

    return inputText

#out = GenerateTextTest(model, TextToTokenIds(txt1, tokeniser), 10, 1024)

###################################################################################### entropy loss stuff
# inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
# targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

# with torch.no_grad():
#     logits = model(inputs)

# probas = torch.softmax(logits, dim=-1)
# tokenIds = torch.argmax(probas, dim=-1, keepdim=True)

# textIdx = 0
# targetProbas1 = probas[textIdx, [0, 1, 2], targets[textIdx]]

# textIdx = 1
# targetProbas2 = probas[textIdx, [0, 1, 2], targets[textIdx]]

# logProbas = torch.log(torch.cat((targetProbas1, targetProbas2)))

# flatLogits =  logits.flatten(0, 1)
# flatTargets = targets.flatten()
# thing = torch.nn.functional.cross_entropy(flatLogits, flatTargets)

######################################################################################

torch.manual_seed(123)
tokeniser = tik.get_encoding("gpt2")

train = pd.read_csv('./Data/CustomerServiceDataSet.csv')
texter = ""

print(train['instruction'][3])
      
numToGet = 500 # len(train['instruction'])-1
for i in range (numToGet):
    texter += (train['instruction'][i] + " ")

charTotal = len(texter) # 1286828
tokenTotal = len(tokeniser.encode(texter)) # 271388

trainRatio = 0.9
splitIdx = int(trainRatio * charTotal)
trainData = texter[:splitIdx]
validationData = texter[splitIdx:]

trainDataloader = GPTDataLoaderClass.CreateDataLoader(trainData, batchSize=2, maxLength=256, stride=256, dropLast=True, shuffleData=True)
valDataLoader = GPTDataLoaderClass.CreateDataLoader(validationData, batchSize=2, maxLength=256, stride=256, dropLast=False, shuffleData=False)

trainTokens = 0
for inBatch, targetBatch in trainDataloader:
    trainTokens += inBatch.numel()

valTokens = 0
for inBatch, targetBatch in valDataLoader:
    valTokens += inBatch.numel()

def calcLossBatch(inputBatch, targetBatch, model):
    logits = model(inputBatch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targetBatch.flatten())
    return loss

def calcLossLoader(dataLoader, model, numBatches=None):
    totalLoss = 0

    if (len(dataLoader) == 0):
        return float("nan")
    elif numBatches is None:
        numBatches = len(dataLoader)
    else:
        numBatches = min(numBatches, len(dataLoader))


    for i, (inputBatch, targetBatch) in enumerate(dataLoader):
        if i < numBatches:
            loss = calcLossBatch(inputBatch, targetBatch, model)
            totalLoss += loss.item()
        else:
            break

    return totalLoss / numBatches

# torch.manual_seed(123)

# with torch.no_grad():
#     trainLoss = calcLossLoader(trainDataloader, model)
#     valLoss = calcLossLoader(valDataLoader, model)

# print(trainLoss)
# print(valLoss)
# print(torch.exp(torch.tensor(trainLoss)))

def EvaluateModel(model, trainLoader, valLoader, evalIter):
    model.eval()
    with torch.no_grad():
        trainLoss = calcLossLoader(trainLoader, model, numBatches=evalIter)
        valLoss = calcLossLoader(valLoader, model, numBatches=evalIter)
    model.train()
    return trainLoss, valLoss

def GenerateAndPrintSample(model, tokeniser, startContext):
    model.eval()
    contextSize = model.positionEmbeddings.weight.shape[0]
    encoded = TextToTokenIds(startContext, tokeniser)

    with torch.no_grad():
        generatedIds = GenerateTextTest(model, encoded, maxNewTokens=50, contextSize=contextSize)
    
    decodedText = TokenIdsToText(generatedIds, tokeniser)
    print(f"Sample Generated Text: {decodedText}")
    model.train()

def TrainModel(model, trainLoader, valLoader, optimiser, numEpochs, evalFreq, evalIter, startContext, tokeniser):
    trainLosses, valLosses, seenTokenTracker = [], [], []
    seenTokens, globalStep = 0 , -1

    for epoch in range(numEpochs):
        model.train()

        for inputBatch, targetBatch in trainLoader:
            optimiser.zero_grad()
            loss = calcLossBatch(inputBatch, targetBatch, model)
            loss.backward()
            optimiser.step()
            seenTokens += inputBatch.numel()
            globalStep += 1

            if globalStep % evalFreq == 0:
                trainLoss, valLoss = EvaluateModel(model, trainLoader, valLoader, evalIter)
                trainLosses.append(trainLoss)
                valLosses.append(valLoss)
                seenTokenTracker.append(seenTokens)
                print(f"Epoch {epoch+1}, Step {globalStep:06d}, Train Loss: {trainLoss:.3f}, Val Loss: {valLoss:.3f}")
        
        #GenerateAndPrintSample(model, tokeniser, startContext)

    return trainLosses, valLosses, seenTokenTracker

model = GPT.GPTModel(GPT.GPT_CONFIG)
optimiser = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
epochNum = 10

trainLosses, valLosses, seenTokenTracker = TrainModel(model, trainDataloader, valDataLoader, optimiser,
                                                          numEpochs=epochNum, evalFreq=5, evalIter=5, 
                                                          startContext="I need to cancel", tokeniser=tokeniser)

def PlotLosses(epochsSeen, tokensSeen, trainLosses, valLosses):
    fig, ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(epochsSeen, trainLosses, label='Training Loss', color='blue')
    ax1.plot(epochsSeen, valLosses, linestyle="-.", label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokensSeen, trainLosses, alpha=0)
    ax2.set_xlabel("Tokens Seen")

    fig.tight_layout()
    plt.show()

epochsTensor = torch.linspace(0, epochNum, len(trainLosses))
PlotLosses(epochsTensor, seenTokenTracker, trainLosses, valLosses)
print("Done")