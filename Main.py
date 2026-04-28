# Library imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import tiktoken as tik
from torch.utils.data import DataLoader
import time

# My class imports
import GPTDataLoaderClass
import GPTModelClass as GPT
from ClassifierTrainingClass import ModelTrainer
from TrainingDataPrepClass import TrainingDataPreper
from ClassifierDataLoaderClass import ClassificationDataset
from TextGenerationClass import TextGeneration
from InstructionTrainerClass import InstructionModelTrainer
from InstructionDatasetLoaderClass import InstructionDataset

######################################################### Initialisation of needed parameters and objects
torch.manual_seed(123)
tokeniser = tik.get_encoding("gpt2") # the tokeniser from the tiktoken library
ClassificationTrainer = ModelTrainer() # my classification training class
instructionTrainer = InstructionModelTrainer() # my instruction training class
dataPreper = TrainingDataPreper() # my data class which reads in the dataset files

######################################################### Model and Optimiser Initialization
model = GPT.GPTModel(GPT.GPT_CONFIG) # initialise the GPT model using the configuration defined in my GPTModelClass.py file

# load the model's trained weights if they exist in the given directory
if os.path.exists("./GPTModel.pth"):
    model.load_state_dict(torch.load("./GPTModel.pth"))
    print("Loaded model weights from GPTModel.pth")
else:
    print("No saved model weights found, starting with a new model")

######################################################### Model Modification for Classification Training only
# # set the number of output dimensions of the model to 2 (same as the number of classifiable intents in the training data)
# numClasses = 2
# model.outHead = torch.nn.Linear(GPT.GPT_CONFIG["embeddingDim"], numClasses)

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
trainDataSet = InstructionDataset('./Data/ProcessedData/TrainData.csv', tokeniser)
trainLoader = DataLoader(trainDataSet, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
for input_batch in trainLoader:
    pass

validDataSet = InstructionDataset('./Data/ProcessedData/ValidData.csv', tokeniser)
validLoader = DataLoader(validDataSet, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
for input_batch in validLoader:
    pass

######################################################### Old training code
# epochNum = 10

# trainLosses, valLosses, seenTokenTracker = Trainer.TrainModel(model, trainDataloader, valDataLoader, optimiser,
#                                                           numEpochs=epochNum, evalFreq=5, evalIter=5, 
#                                                           startContext="I need to cancel", tokeniser=tokeniser)

# def PlotLosses(epochsSeen, tokensSeen, trainLosses, valLosses):
#     fig, ax1 = plt.subplots(figsize=(5,3))

#     ax1.plot(epochsSeen, trainLosses, label='Training Loss', color='blue')
#     ax1.plot(epochsSeen, valLosses, linestyle="-.", label='Validation Loss', color='orange')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss')
#     ax1.legend(loc = "upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

#     ax2 = ax1.twiny()
#     ax2.plot(tokensSeen, trainLosses, alpha=0)
#     ax2.set_xlabel("Tokens Seen")

#     fig.tight_layout()
#     plt.show()

# epochsTensor = torch.linspace(0, epochNum, len(trainLosses))
# print("Done Training")

# PlotLosses(epochsTensor, seenTokenTracker, trainLosses, valLosses)

# torch.save(model.state_dict(), "./GPTModel.pth")
# print("Done and saved training")


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