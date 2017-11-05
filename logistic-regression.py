#Logistic regression: Classify inputs into classes

# Training:
# 1. Weight Matrix Multiplication
# 2. Bias addition

# 3. Fitting to Sigmoid probability curve

# Implement 3 function model as:
# tf.matmul(X, weights) - matrix Multiplication. Takes input and weights as paramaters
# tf.add(weighted_X, bias)
# tf.nn.sigmoid(weighted_X_with_bias)

import tensorflow as tf
import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
