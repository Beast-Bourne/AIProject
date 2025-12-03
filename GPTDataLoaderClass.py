from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken as tik
        
class GPTDataSet(Dataset):
    def __init__(self, text, tokenizer, maxLength, stride):
        self.inputIds = []
        self.targetIds = []

        # tokenise the entire given text using the tokensizer
        tokenIds = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})

        # chunk the text using a sliding window into overlapping sequences of length 'maxLength'
        for i in range(0, len(tokenIds)-maxLength, stride):
            inputChunk = tokenIds[i: i + maxLength]
            targetChunk = tokenIds[i+1: i + maxLength + 1]
            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))
    
    def __len__(self):
        return len(self.inputIds)
    
    def __getitem__(self, idx):
        return self.inputIds[idx], self.targetIds[idx]
    
def CreateDataLoader(text, batchSize=4, maxLength=256, stride=128, shuffleData=True, dropLast=True, numOfWorkers=0):
    tokenizer= tik.get_encoding("gpt2")

    dataset = GPTDataSet(text, tokenizer, maxLength, stride)

    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffleData, drop_last=dropLast, num_workers=numOfWorkers)
    return dataloader
