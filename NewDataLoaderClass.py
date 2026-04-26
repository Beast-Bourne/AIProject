import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csvFilePath, tokeniser, maxLength=None, padToken = 50256):
        self.data = pd.read_csv(csvFilePath)

        self.encodedTexts = [tokeniser.encode(text) for text in self.data['instruction']]

        # truncate the encoded texts to the maxLength
        if maxLength is None:
            self.maxLength = max(len(encoded) for encoded in self.encodedTexts)
            self.maxLength += 1 # add 1 to account for the end of sequence token
        else:
            self.maxLength = maxLength
        self.encodedTexts = [encodedText[:self.maxLength] for encodedText in self.encodedTexts]

        # pad the encoded texts with the 'end of text' token to ensure they all have the same length
        self.encodedTexts = [encodedText + [padToken] * (self.maxLength - len(encodedText)) for encodedText in self.encodedTexts]
        

    def __len__(self):
        return len(self.data)

    # this could be improved by adding a return for the intent of the instruction
    def __getitem__(self, idx):
        encoded = self.encodedTexts[idx]
        #label = self.data.iloc[idx]["intent"]
        return torch.tensor(encoded, dtype=torch.long)