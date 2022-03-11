import csv
import torch
import math
from sklearn.model_selection import train_test_split

# Himmelblau test function (imput/output normalized)
def himmelblau(gamma, llambda):
    # input scaling
    gamma = gamma * 5
    llambda = llambda * 5

    # function
    score = (torch.square(torch.square(gamma) + llambda - 11) + torch.square(gamma + torch.square(llambda) - 7))

    # output scaling
    return score * 0.001184


