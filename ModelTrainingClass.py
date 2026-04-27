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
    
    def CalcAccuracyLoader(self, dataLoader, model, numBatches=None):
        model.eval()
        correctPreds, numPreds = 0, 0

        if numBatches is None:
            numBatches = len(dataLoader)
        else:
            numBatches = min(numBatches, len(dataLoader))

        for i, (inputBatch, targetBatch) in enumerate(dataLoader):
            if i < numBatches:
                with torch.no_grad():
                    logits = model(inputBatch)[:, -1, :]
                predictions = torch.argmax(logits, dim=-1)

                numPreds += predictions.shape[0]
                correctPreds += (predictions == targetBatch).sum().item()
            else:
                break
        
        return correctPreds / numPreds

    # Trains the model for a specified number of epochs, iterating through the training data loader and updating the model's parameters using the provided optimiser. 
    # The method also evaluates the model's performance at regular intervals defined by evalFreq, and tracks the training and validation losses and the number of tokens seen during training.\
    # Additionally, it generates and prints a sample of text after each epoch using the TextGeneration class.
    def TrainModel(self, model, trainLoader, valLoader, optimiser, numEpochs, evalFreq, evalIter, startContext, tokeniser):
        trainLosses, valLosses, trainAccur, valAccur = [], [], [], []
        seenExamples, globalStep = 0 , -1

        # main training loop
        for epoch in range(numEpochs):
            model.train()

            # iterate through batches of training data
            for inputBatch, targetBatch in trainLoader:
                optimiser.zero_grad() # reset loss gradient
                loss = self.CalcLossBatch(inputBatch, targetBatch, model)
                loss.backward() # calculate loss gradient
                optimiser.step() # update model weights based on loss gradient
                seenExamples += inputBatch.shape[0] # track number of examples seen during training
                globalStep += 1

                # evaluate model performance at regular intervals defined by evalFreq
                if globalStep % evalFreq == 0:
                    trainLoss, valLoss = self.EvaluateModel(model, trainLoader, valLoader, evalIter)
                    trainLosses.append(trainLoss)
                    valLosses.append(valLoss)
                    print(f"Epoch {epoch+1}, Step {globalStep:06d}, Train Loss: {trainLoss:.3f}, Val Loss: {valLoss:.3f}")
            
            # generate and print a sample of text after each epoch using the TextGeneration class
            self.textGen.GenerateAndPrintSample(model, tokeniser, startContext)

            # Calculate accuracy after each epoch
            tAccuracy = self.CalcAccuracyLoader(trainLoader, model, numBatches=evalIter)
            vAccuracy = self.CalcAccuracyLoader(valLoader, model, numBatches=evalIter)
            print(f"Epoch {epoch+1}, Train Accuracy: {tAccuracy:.3f}, Val Accuracy: {vAccuracy:.3f}")
            trainAccur.append(tAccuracy)
            valAccur.append(vAccuracy)

        return trainLosses, valLosses, trainAccur, valAccur, seenExamples