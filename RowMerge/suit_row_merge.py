import csv
import os
import random
import math
import re

import suit_sparse_matrix


    
if __name__ == "__main__":

    # suit_sparse_matrix.generate_suitesparse_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v8/", "./suitpaths.txt", 8)
    # suit_sparse_matrix.generate_suitesparse_CVSE_file("../benchmark/dlmc/", "../benchmark/dlmc-v16/", "./suitpaths.txt", 16)
    
    suit_sparse_matrix.generate_suitesparse_csv_file("../benchmark/dlmc-v8/", "./suitpaths.txt", 8)
    suit_sparse_matrix.generate_suitesparse_csv_file("../benchmark/dlmc-v16/", "./suitpaths.txt", 16)
    
    
    ##################################################################################################
    # m0 = suit_csr("../benchmark/dlmc/suitesparse/ASIC_680k_csr.mtx")
    # m1 = suit_sparse_matrix.suit_cvse("../benchmark/dlmc-v8/suitesparse/in-2004.smtx", 8)
    # print(m1.new_sparsity())
    # print(m1.all_zero_vec_cnt())
    # print(m1.return_CSR_storage())
    # print(m1.return_CVSE_storage())
    # print(m1.return_TCF_storage())
    # print(m1.return_SRBCSR_storage())
