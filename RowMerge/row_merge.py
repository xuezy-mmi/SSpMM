import csv
import os
import random
from math import log
# import matplotlib.pyplot as plt

import sparse_matrix_format
# import reorder


if __name__ == "__main__":
    #################################################################################
    #############################directly merge######################################
    #################################################################################
    #############################     V = 8    ######################################
    sparse_matrix_format.generate_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v8/",  "./tfpaths.txt", 8)
    #############################     V =16    ######################################
    sparse_matrix_format.generate_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v16/", "./tfpaths.txt", 16)
    
    # #############################     V = 8    ######################################
    # sparse_matrix_format.generate_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v8/",  "./suitpaths.txt", 8)
    # #############################     V =16    ######################################
    # sparse_matrix_format.generate_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v16/", "./suitpaths.txt", 16)
    # #################################################################################
    # #################################################################################
    # #################################################################################