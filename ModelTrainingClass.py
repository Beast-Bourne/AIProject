import torch
from TextGenerationClass import TextGeneration

class ModelTrainer:
    def __init__(self):
        self.textGen = TextGeneration()

    # Calculates the cross-entropy loss for a batch of input and target data using the provided model
    def CalcLossBatch(self, inputBatch, targetBatch, model):
        logits = model(inputBatch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, targetBatch.flatten())
        return loss

    # Calculates the average loss over a data loader by iterating through batches and using the CalcLossBatch method, with an option to limit the number of batches evaluated
    def CalcLossLoader(self, dataLoader, model, numBatches=None):
        totalLoss = 0

        if (len(dataLoader) == 0):
            return float("nan")
        elif numBatches is None:
            numBatches = len(dataLoader)
        else:
            numBatches = min(numBatches, len(dataLoader))


        for i, (inputBatch, targetBatch) in enumerate(dataLoader):
            if i < numBatches:
                loss = self.CalcLossBatch(inputBatch, targetBatch, model)
                totalLoss += loss.item()
            else:
                break

        return totalLoss / numBatches
    
    # Evaluates the model's performance on both the training and validation data loaders by calculating the average loss for a specified number of batches, and returns the training and validation losses
    def EvaluateModel(self, model, trainLoader, valLoader, evalIter):
        model.eval()
        with torch.no_grad():
            trainLoss = self.CalcLossLoader(trainLoader, model, numBatches=evalIter)
            valLoss = self.CalcLossLoader(valLoader, model, numBatches=evalIter)
        model.train()
        return trainLoss, valLoss

    # Trains the model for a specified number of epochs, iterating through the training data loader and updating the model's parameters using the provided optimiser. 
    # The method also evaluates the model's performance at regular intervals defined by evalFreq, and tracks the training and validation losses and the number of tokens seen during training.\
    # Additionally, it generates and prints a sample of text after each epoch using the TextGeneration class.
    def TrainModel(self, model, trainLoader, valLoader, optimiser, numEpochs, evalFreq, evalIter, startContext, tokeniser):
        trainLosses, valLosses, seenTokenTracker = [], [], []
        seenTokens, globalStep = 0 , -1

        for epoch in range(numEpochs):
            model.train()

            for inputBatch, targetBatch in trainLoader:
                optimiser.zero_grad()
                loss = self.CalcLossBatch(inputBatch, targetBatch, model)
                loss.backward()
                optimiser.step()
                seenTokens += inputBatch.numel()
                globalStep += 1

                if globalStep % evalFreq == 0:
                    trainLoss, valLoss = self.EvaluateModel(model, trainLoader, valLoader, evalIter)
                    trainLosses.append(trainLoss)
                    valLosses.append(valLoss)
                    seenTokenTracker.append(seenTokens)
                    print(f"Epoch {epoch+1}, Step {globalStep:06d}, Train Loss: {trainLoss:.3f}, Val Loss: {valLoss:.3f}")
            
            self.textGen.GenerateAndPrintSample(model, tokeniser, startContext)

        return trainLosses, valLosses, seenTokenTracker