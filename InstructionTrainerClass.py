import torch
import pandas as pd
from InstructionTextGeneratorClass import InstructionTextGeneration

def FormatInstructionInput(input):
    instructionText = (f"Below is an instruction that describes a task."
                       f"Write a response that appropriately completes the request."
                       f"\n\n### Instruction:\n{input}"
                       f"\n\n### Response:")
    return instructionText

class InstructionModelTrainer:
    def __init__(self):
        self.textGen = InstructionTextGeneration()

    # Calculates the cross-entropy loss for a batch of input and target data using the provided model
    def CalcLossBatch(self, inputBatch, targetBatch, model):
        logits = model(inputBatch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), targetBatch.flatten())
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
    def TrainModel(self, model, trainLoader, valLoader, optimiser, numEpochs, evalFreq, evalIter, testText1, testText2, tokeniser):
        trainLosses, valLosses, trackTokensSeen = [], [], []
        tokensSeen, globalStep = 0 , -1

        testInput1 = FormatInstructionInput(testText1)
        testInput2 = FormatInstructionInput(testText2)

        epochData = []

        # main training loop
        for epoch in range(numEpochs):
            model.train()

            # iterate through batches of training data
            for inputBatch, targetBatch in trainLoader:
                optimiser.zero_grad() # reset loss gradient
                loss = self.CalcLossBatch(inputBatch, targetBatch, model)
                loss.backward() # calculate loss gradient
                optimiser.step() # update model weights based on loss gradient
                tokensSeen += inputBatch.numel() # track the number of tokens seen during training
                globalStep += 1

                # evaluate model performance at regular intervals defined by evalFreq
                if globalStep % evalFreq == 0:
                    trainLoss, valLoss = self.EvaluateModel(model, trainLoader, valLoader, evalIter)
                    trainLosses.append(trainLoss)
                    valLosses.append(valLoss)
                    trackTokensSeen.append(tokensSeen)
                    print(f"Epoch {epoch+1}, Step {globalStep:06d}, Train Loss: {trainLoss:.3f}, Val Loss: {valLoss:.3f}")
            
            # generate and print a sample of text after each epoch using the TextGeneration class
            #print(testInput1)
            modelOutput1 = self.textGen.GenerateAndPrintSample(model, tokeniser, testInput1, printText=False)

            #print("\n\n", testInput2)
            modelOutput2 = self.textGen.GenerateAndPrintSample(model, tokeniser, testInput2, printText=False)

            epochData.append({
                "Instruction1": testText1,
                "ModelOutput1": modelOutput1,
                "Instruction2": testText2,
                "ModelOutput2": modelOutput2,
            })

        outputDataFrame = pd.DataFrame(epochData, columns=["Instruction1", "ModelOutput1", "Instruction2", "ModelOutput2"])

        return trainLosses, valLosses, trackTokensSeen, outputDataFrame