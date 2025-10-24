import numpy as np
import pandas as pd
import tensorflow as tf
import torch as tch

from linkTest import Test


print(tch.__version__)
print(tch.cuda.is_available())
print(tf.__version__)
Test.testFunc()