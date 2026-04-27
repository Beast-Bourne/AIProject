import pandas as pd
import os

class TrainingDataPreper:
    def __init__(self, numSamples=600, randSeed=123):
        self.randSeed = randSeed
        self.numSamples = numSamples
        self.rawData = pd.read_csv('./Data/CustomerServiceDataSet.csv')

        # if the processed data files already exist then just read the data from them
        if os.path.exists("./Data/ProcessedData/TrainData.csv") and \
           os.path.exists("./Data/ProcessedData/ValidData.csv") and \
           os.path.exists("./Data/ProcessedData/TestData.csv"):
            self.trainData, self.validData, self.testData = self.LoadDataFromFile("./Data/ProcessedData/TrainData.csv", 
                                                                  "./Data/ProcessedData/ValidData.csv", 
                                                                  "./Data/ProcessedData/TestData.csv")
            print("Loaded processed data from files")
        
        # otherwise split the data and save the splits to files for future use
        else:
            self.trainData, self.validData, self.testData = self.SplitAndSaveDataFromIntent(0.8, 0.1)
            os.makedirs('./Data/ProcessedData', exist_ok=True)
            self.trainData.to_csv('./Data/ProcessedData/TrainData.csv', index=None)
            self.validData.to_csv('./Data/ProcessedData/ValidData.csv', index=None)
            self.testData.to_csv('./Data/ProcessedData/TestData.csv', index=None)
            print("Processed data not found, created new splits and saved to files")

    # this function takes a random sample of data from the dataset for the given intent
    # it shuffles and splits the data into training, validation and test sets
    def SplitAndSaveDataFromIntent(self, trainRatio, validRatio):
        sampleNum = int(self.numSamples/2)
        dataSet1 = self.rawData[self.rawData["intent"] == "cancel_order"].sample(sampleNum, random_state=self.randSeed)
        dataSet2 = self.rawData[self.rawData["intent"] == "place_order"].sample(sampleNum, random_state=self.randSeed)
        dataSet = pd.concat([dataSet1, dataSet2], ignore_index=True)

        shuffledData = dataSet.sample(frac=1, random_state=self.randSeed).reset_index(drop=True)

        trainEnd = int(len(shuffledData) * trainRatio)
        validEnd = trainEnd + int(len(shuffledData) * validRatio)

        trainData = shuffledData[:trainEnd] # this data is used for training the model and updating its weights
        validData = shuffledData[trainEnd:validEnd] # this data is used for evaluating the model's performance during training, but not for updating the model's weights
        testData = shuffledData[validEnd:] # this data is used for evaluating the model's performance after training is complete, and is not used during the training process at all

        return trainData, validData, testData
    
    # this function reads the training, validation and test data from the given file paths and returns them
    def LoadDataFromFile(self, TrainPath, ValidPath, TestPath):
        trainData = pd.read_csv(TrainPath)
        validData = pd.read_csv(ValidPath)
        testData = pd.read_csv(TestPath)

        return trainData, validData, testData
        