import pandas as pd
import os

class TrainingDataPreper:
    def __init__(self, numSamples=500, randSeed=123):
        self.randSeed = randSeed
        self.numSamples = numSamples
        self.allData = pd.read_csv('./Data/CustomerServiceDataSet.csv')

        if os.path.exists("./Data/ProcessedData/TrainData.csv") and \
           os.path.exists("./Data/ProcessedData/ValidData.csv") and \
           os.path.exists("./Data/ProcessedData/TestData.csv"):
            self.trainData, self.validData, self.testData = self.LoadDataFromFile("./Data/ProcessedData/TrainData.csv", 
                                                                  "./Data/ProcessedData/ValidData.csv", 
                                                                  "./Data/ProcessedData/TestData.csv")
            print("Loaded processed data from files")
            
        else:
            self.trainData, self.validData, self.testData = self.SplitAndSaveDataFromIntent("cancel_order", 0.7, 0.1)
            os.makedirs('./Data/ProcessedData', exist_ok=True)
            self.trainData.to_csv('./Data/ProcessedData/TrainData.csv', index=None)
            self.validData.to_csv('./Data/ProcessedData/ValidData.csv', index=None)
            self.testData.to_csv('./Data/ProcessedData/TestData.csv', index=None)
            print("Processed data not found, created new splits and saved to files")

    def SplitAndSaveDataFromIntent(self, Intent, trainRatio, validRatio):
        dataSet = self.allData[self.allData["intent"] == Intent].sample(self.numSamples, random_state=self.randSeed)

        shuffledData = dataSet.sample(frac=1, random_state=self.randSeed).reset_index(drop=True)

        trainEnd = int(len(shuffledData) * trainRatio)
        validEnd = trainEnd + int(len(shuffledData) * validRatio)

        trainData = shuffledData[:trainEnd] # this data is used for training the model and updating its weights
        validData = shuffledData[trainEnd:validEnd] # this data is used for evaluating the model's performance during training, but not for updating the model's weights
        testData = shuffledData[validEnd:] # this data is used for evaluating the model's performance after training is complete, and is not used during the training process at all

        return trainData, validData, testData
    
    def LoadDataFromFile(self, TrainPath, ValidPath, TestPath):
        trainData = pd.read_csv(TrainPath)
        validData = pd.read_csv(ValidPath)
        testData = pd.read_csv(TestPath)

        return trainData, validData, testData
        