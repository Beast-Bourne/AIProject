import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial

# formats the input instruction for the LLM
def FormatInput(input):
    instructionText = (f"Below is an instruction that describes a task."
                       f"Write a response that appropriately completes the request."
                       f"\n\n### Instruction:\n{input}")
    return instructionText

def CollateDraft(batch, padToken=50256, ignoreToken=-100, allowedMaxLength=None):
    batchMaxLength = max(len(item)+1 for item in batch)
    inputList, targetList = [], []

    for item in batch:
        newItem = item.copy()
        newItem += [padToken] # add a single 'end of text' token
        padded = (newItem + [padToken] * (batchMaxLength - len(newItem))) # pad all items to the 'batchMaxLength'

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # replaces all but the fist padding token with the ignore token (loss calculation will ignore these tokens)
        mask = targets == padToken
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignoreToken

        # truncates the inputs and targets to the allowed maximum length
        if allowedMaxLength is not None:
            inputs = inputs[:allowedMaxLength]
            targets = targets[:allowedMaxLength]

        inputList.append(inputs)
        targetList.append(targets)
    
    # convert list of inputs to tensors
    inputTensors = torch.stack(inputList)
    targetTensors = torch.stack(targetList)
    return inputTensors, targetTensors

class InstructionDataset(Dataset):
    def __init__(self, csvFilePath, tokeniser):
        self.data = pd.read_csv(csvFilePath)

        self.encodedTexts = []
        for i in range(len(self.data)):
            instructionText = FormatInput(self.data.iloc[i]["instruction"])
            responseText = f"\n\n### Response:\n{self.data.iloc[i]['response']}"
            fullText = instructionText + responseText
            self.encodedTexts.append(tokeniser.encode(fullText))

    def __len__(self):
        return len(self.data)

    # this could be improved by adding a return for the intent of the instruction
    def __getitem__(self, idx):
        return self.encodedTexts[idx]
    
# the collate function for the instruction dataset, which will be used in the DataLoader to format the batches of data
# this is where to modify the parameters of the collate function (padding token, ignore token, allowed maximum length)
customCollateFunc = partial(CollateDraft)

# function to create the DataLoader for the instruction dataset, which will be used in the training loop
def GetInstructionDataLoader(dataFilePath, tokeniser, batchSize=8, shuffle=True, dropLast=True, numWorkers=0):
    dataset = InstructionDataset(dataFilePath, tokeniser)
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, drop_last=dropLast, num_workers=numWorkers, collate_fn=customCollateFunc)
    return dataLoader