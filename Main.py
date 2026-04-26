# Library imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import tiktoken as tik

# My class imports
import GPTDataLoaderClass
import GPTModelClass as GPT
from ModelTrainingClass import ModelTrainer
from TrainingDataPrepClass import TrainingDataPreper

######################################################### Initialisation of needed parameters and objects
torch.manual_seed(123)
tokeniser = tik.get_encoding("gpt2") # the tokeniser from the tiktoken library
Trainer = ModelTrainer() # my model training class

model = GPT.GPTModel(GPT.GPT_CONFIG) # initialise the GPT model using the configuration defined in my GPTModelClass.py file

# load the model's trained weights if they exist in the given directory
if os.path.exists("./GPTModel.pth"):
    model.load_state_dict(torch.load("./GPTModel.pth"))
    print("Loaded model weights from GPTModel.pth")
else:
    print("No saved model weights found, starting with a new model")

optimiser = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # AdamW optimiser from PyTorch with a learning rate of 0.0004 and weight decay of 0.1

train = pd.read_csv('./Data/CustomerServiceDataSet.csv') # read in the dataset


######################################################### Data Preparation and DataLoader Creation
texter = ""
numToGet = len(train['instruction'])-1

for i in range (numToGet):
    if train["intent"][i] == "cancel_order":
        texter += (train['instruction'][i] + " ")

charTotal = len(texter) # 1286828 for full dataset
tokenTotal = len(tokeniser.encode(texter)) # 271388 for full dataset

trainRatio = 0.9
splitIdx = int(trainRatio * charTotal)
trainData = texter[:splitIdx]
validationData = texter[splitIdx:]

trainDataloader = GPTDataLoaderClass.CreateDataLoader(trainData, batchSize=2, maxLength=256, stride=256, dropLast=True, shuffleData=True)
valDataLoader = GPTDataLoaderClass.CreateDataLoader(validationData, batchSize=2, maxLength=256, stride=256, dropLast=False, shuffleData=False)


######################################################### Training the Model and Plotting Losses
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


######################################################### Testing things

#print(train["intent"].value_counts())
#print(tokeniser.decode([50256]))

dataPreper = TrainingDataPreper(tokeniser)
print(dataPreper.trainData.head())