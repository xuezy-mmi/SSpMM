import math
import numpy as np
import pandas as pd
import csv

def Fmean(array):
    len_a = len(array)
    sum_a = sum(array)
    return float(sum_a / len_a)

def read_perf(path):
    csvfile = pd.read_csv(path,usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
    filenum = len(csvfile.nnz)
    validnum = 0
    dis_perf = [[] for i in range(7)]
    dis_mean = [[] for i in range(7)]
    total_mean = 0
    for i in range(4):
        for j in range(7):
            # xlabel = sp[j]
            for k in range(96):
                NO_file = i*7*96+j*96+k
                if(int(csvfile.nnz[NO_file])!=0):
                    dis_perf[j].append(float(csvfile.perf[NO_file]))
                    validnum += 1
    # print(filenum)
    # print(validnum)
    # print(len(dis_perf))
    # print(len(dis_perf[0]), len(dis_perf[1]), len(dis_perf[2]), len(dis_perf[3]), len(dis_perf[4]), len(dis_perf[5]), len(dis_perf[6]))
    for i in range(7):
        dis_mean[i] = Fmean(dis_perf[i])
        total_mean += sum(dis_perf[i])
    total_mean = total_mean / validnum
    # print(dis_mean)
    # print(total_mean)
    return total_mean, dis_mean

def norm_perf_list(perf, mean_perf):
    for i in range(7):
        list_len = len(perf[i])
        for j in range(list_len):
            perf[i][j] = perf[i][j] / mean_perf[i]
    return perf

def norm_perf(dis_mean, mean_perf):
    for i in range(7):
        dis_mean[i] = dis_mean[i] / mean_perf[i]
    return dis_mean

if __name__ == "__main__":
    # csvfile = pd.read_csv('../result-data/ampere_cublas.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
    # filenum = len(csvfile.nnz)
    # validnum = 0
    # dis_perf = [[] for i in range(7)]
    # dis_mean = [[] for i in range(7)]
    # total_mean = 0
    # for i in range(4):
    #     for j in range(7):
    #         # xlabel = sp[j]
    #         for k in range(96):
    #             NO_file = i*7*96+j*96+k
    #             if(int(csvfile.nnz[NO_file])!=0):
    #                 dis_perf[j].append(float(csvfile.perf[NO_file]))
    #                 validnum += 1
                    
    # print(filenum)
    # print(validnum)
    # print(len(dis_perf))
    # print(len(dis_perf[0]), len(dis_perf[1]), len(dis_perf[2]), len(dis_perf[3]), len(dis_perf[4]), len(dis_perf[5]), len(dis_perf[6]))

    # for i in range(7):
    #     dis_mean[i] = Fmean(dis_perf[i])
    #     total_mean += sum(dis_perf[i])
    # total_mean = total_mean / validnum
    # print(dis_mean)
    # print(total_mean)
    path1 = "../result-data/ampere_cublas.csv"
    total_mean1, dis_mean1 = read_perf(path1)
    print("cublas total mean perf: ", total_mean1)
    print(dis_mean1)
    path2 = "../result-data/ampere_cusparse.csv"
    total_mean2, dis_mean2 = read_perf(path2)
    print("cusparse total mean perf: ", total_mean2)
    print(dis_mean2)