import numpy as np
import pandas as pd
import re
import tensorflow as tf
import torch as tch

from TokeniserClass import Tokeniser

train = pd.read_csv('./Data/CustomerServiceDataSet.csv')

tokeniserRef = Tokeniser(train)

texter = train['instruction'][5001]
print(texter)

test = tokeniserRef.TokeniseText(texter)
print(test)

test2 = tokeniserRef.DetokeniseArray(test)
print(test2)

