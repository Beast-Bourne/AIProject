# Library imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import tiktoken as tik
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# My class imports
import GPTDataLoaderClass
import GPTModelClass as GPT
from ClassifierTrainingClass import ModelTrainer
from TrainingDataPrepClass import TrainingDataPreper
from ClassifierDataLoaderClass import ClassificationDataset
from TextGenerationClass import TextGeneration
from InstructionTrainerClass import InstructionModelTrainer
from InstructionDatasetLoaderClass import GetInstructionDataLoader
from InstructionTextGeneratorClass import InstructionTextGeneration

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

######################################################### Initialisation of needed parameters and objects
ModelInfo = "300Samples1Epochs"

torch.manual_seed(123)
tokeniser = tik.get_encoding("gpt2") # the tokeniser from the tiktoken library
ClassificationTrainer = ModelTrainer() # my classification training class
instructionTrainer = InstructionModelTrainer() # my instruction training class
dataPreper = TrainingDataPreper() # my data class which reads in the dataset files

######################################################### Model and Optimiser Initialization
model = GPT.GPTModel(GPT.GPT_CONFIG) # initialise the GPT model using the configuration defined in my GPTModelClass.py file

######################################################### Model Modification for Classification Training only
# # set the number of output dimensions of the model to 2 (same as the number of classifiable intents in the training data)
# numClasses = 2
# model.outHead = torch.nn.Linear(GPT.GPT_CONFIG["embeddingDim"], numClasses)

######################################################### Model Modification for Instruction Training only
model.outHead = torch.nn.Linear(GPT.GPT_CONFIG["embeddingDim"], GPT.GPT_CONFIG["vocabSize"])

######################################################### model loading from save file
# load the model's trained weights if they exist in the given directory
modelPath = "./Models/GPTModel" + ModelInfo + ".pth"

os.makedirs('./Models', exist_ok=True)
if os.path.exists(modelPath):
    model.load_state_dict(torch.load(modelPath))
    print("Loaded model weights from GPTModel.pth")
else:
    print("No saved model weights found, starting with a new model")

######################################################### model modification for fine-tuning
for param in model.transformerBlocks.parameters():
    param.requires_grad = True
for param in model.finalNorm.parameters():
    param.requires_grad = True

optimiser = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) # AdamW optimiser from PyTorch

######################################################### Classification DataLoader Creation
# trainDataSet = ClassificationDataset('./Data/ProcessedData/TrainData.csv', tokeniser)
# trainLoader = DataLoader(trainDataSet, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
# for input_batch in trainLoader:
#     pass

# validDataSet = ClassificationDataset('./Data/ProcessedData/ValidData.csv', tokeniser)
# validLoader = DataLoader(validDataSet, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
# for input_batch in validLoader:
#     pass

######################################################### Instruction DataLoader Creation
trainLoader = GetInstructionDataLoader('./Data/ProcessedData/TrainData.csv', tokeniser, batchSize=8, shuffle=True, dropLast=True, numWorkers=0)
validLoader = GetInstructionDataLoader('./Data/ProcessedData/ValidData.csv', tokeniser, batchSize=8, shuffle=False, dropLast=False, numWorkers=0)
testLoader = GetInstructionDataLoader('./Data/ProcessedData/TestData.csv', tokeniser, batchSize=8, shuffle=False, dropLast=False, numWorkers=0)

######################################################### Classification training
# startTime = time.time()

# numEpochs = 5
# trainLosses, valLosses, trainAccs, valAccs, examplesSeen = ClassificationTrainer.TrainModel(
#     model, trainLoader, validLoader, optimiser, numEpochs, 
#     evalFreq=50, evalIter=5, startContext="I need to cancel", tokeniser=tokeniser)

# endTime = time.time()
# executetime = (endTime - startTime)/60
# print(f"Done Training, execution time: {executetime:.2f} minutes")

######################################################### Instruction training
startTime = time.time()

inputText1 = "I need to cancel my order"
inputText2 = "I want to place an order"

numOfEpochs = 1
trainLosses, valLosses, tokensSeen, modelOutputData = instructionTrainer.TrainModel(
    model, trainLoader, validLoader, optimiser, 
    numEpochs=numOfEpochs, evalFreq=50, evalIter=5, 
    testText1=inputText1, testText2=inputText2, tokeniser=tokeniser)

endTime = time.time()
executetime = (endTime - startTime)/60
print(f"\n\nDone Training, execution time: {executetime:.2f} minutes")

torch.save(model.state_dict(), modelPath)
print("\nDone and saved training")

modelDataPath = "./Data/ProcessedData/ModelResponses" + ModelInfo + ".csv"
modelOutputData.to_csv(modelDataPath, index=None)

epochTensors = torch.linspace(0, numOfEpochs, len(trainLosses))
PlotLosses(epochTensors, tokensSeen, trainLosses, valLosses)