import math
import numpy as np
import pandas as pd
import csv

def Fmean(array):
    len_a = len(array)
    sum_a = sum(array)
    return float(sum_a / len_a)

def read_perf_SSpMM(path1, path2):
    csvfile1 = pd.read_csv(path1,usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
    csvfile2 = pd.read_csv(path2,usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
    filenum = 96*7*4
    validnum = 0
    dis_perf = [[] for i in range(7)]
    dis_mean = [[] for i in range(7)]
    total_mean = 0
    for i in range(4):
        for j in range(7):
            # xlabel = sp[j]
            for k in range(96):
                NO_file = i*7*96+j*96+k
                if(int(csvfile1.nnz[NO_file])!=0):
                    perf1 = float(csvfile1.perf[NO_file])
                    perf2 = float(csvfile2.perf[NO_file])
                    if(perf2 > perf1):
                        perf1 = perf2
                    dis_perf[j].append(perf1)
                    validnum += 1
    for i in range(7):
        dis_mean[i] = Fmean(dis_perf[i])
        total_mean += sum(dis_perf[i])
    total_mean = total_mean / validnum
    return total_mean, dis_mean, dis_perf

def read_perf(path):
    csvfile = pd.read_csv(path,usecols=[0,2],names=["nnz", "perf"])
    filenum = 2688
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
    for i in range(7):
        dis_mean[i] = Fmean(dis_perf[i])
        total_mean += sum(dis_perf[i])
    total_mean = total_mean / validnum
    return total_mean, dis_mean, dis_perf

def read_perf4(path):
    csvfile = pd.read_csv(path,usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
    filenum = 96*7*4
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
                    # else:
                        # print(csvfile.M[NO_file])
    for i in range(7):
        dis_mean[i] = Fmean(dis_perf[i])
        total_mean += sum(dis_perf[i])
    total_mean = total_mean / validnum
    return total_mean, dis_mean, dis_perf

def norm_perf(dis_perf, dis_mean):
    norm_perf = [[] for i in range(7)]
    for i in range(7):
        list_len = len(dis_perf[i])
        for j in range(list_len):
            norm_perf[i].append(float(dis_perf[i][j] / dis_mean[i]))
    return norm_perf
        
def read_suit_perf(path):
    csvfile = pd.read_csv(path,usecols=[2],names=["perf"])
    filenum = len(csvfile.perf)
    dis_perf = [0.0 for i in range(15)]
    for i in range(filenum):
        dis_perf[i] = float(csvfile.perf[i])
    return dis_perf

def read_suit_SSpMMperf(path1, path2):
    csvfile1 = pd.read_csv(path1,usecols=[4],names=["perf"])
    csvfile2 = pd.read_csv(path2,usecols=[4],names=["perf"])
    filenum = len(csvfile1.perf)
    dis_perf = [0.0 for i in range(15)]
    for i in range(filenum):
        perf1 = float(csvfile1.perf[i])
        perf2 = float(csvfile2.perf[i])
        if(perf2 > perf1):
            perf1 = perf2
        dis_perf[i] = perf1
    return dis_perf

if __name__ == "__main__":
    # csvfile = pd.read_csv('../../result-data/ampere_cublas.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
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
    
    # path = "../../result-data/ampere_cublas.csv"
    # total_mean, dis_mean, dis_perf = read_perf96(path)
    # path1 = "../../data/ada_rospmm_16_v0.csv"
    # total_mean1, dis_mean1, dis_perf1 = read_perf97(path1)
    # norm_perf = norm_perf(dis_perf1, dis_mean1)
    # print(len(dis_mean1), len(dis_perf1[6]), len(norm_perf[6]))
    path = "./ampere_wmmaspmm_8_v0.csv"
    total_mean, dis_mean, dis_perf = read_perf96(path)
    path1 = "./ampere_wmmaspmm_16_v0.csv"
    total_mean1, dis_mean1, dis_perf1 = read_perf96(path1)
    print("wmma_spmm_v8")
    print(total_mean)
    print(dis_mean)
    print("wmma_spmm_v16")
    print(total_mean1)
    print(dis_mean1)
    path = "./ampere_rospmm_8_v0.csv"
    total_mean2, dis_mean2, dis_perf2 = read_perf96(path)
    path1 = "./ampere_rospmm_16_v0.csv"
    total_mean3, dis_mean3, dis_perf3 = read_perf96(path1)
    print("ro_spmm_v8")
    print(total_mean2)
    print(dis_mean2)
    print("ro_spmm_v16")
    print(total_mean3)
    print(dis_mean3)