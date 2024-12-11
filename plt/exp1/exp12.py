import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker
import seaborn
# 0 matrix 
# 1 nnzA
# 2 dasp_gflops
# 3 cusparse
# 4 csr5
# 5 tile
# 6 bsr
# 7 lsrb
# 8 dasp/cupsarse
# 9 dasp/csr5
# 10 dasp/tile
# 11 dasp/bsr
# 12 dasp/lsrb

# tile_com_01=pd.read_csv('spmv_fp64_5_method.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12],names=['compute_nnz','per_dasp','per_cusp','per_csr5','per_tile','per_bsr','per_lsrb','sp_cusp','sp_csr5','sp_tile','sp_bsr','sp_lsrb'])

csvfile1=pd.read_csv('../../result-data/RowMerge_V8.csv',usecols=[1,2,3,4,5,6,7,8,9], names=["ROW","COL","NNZ","new_sparsity", "all_zero_vec_cnt", "CSR_storage", "CVSE_storage", "TCF_storage", "SR_BCSR_storage"])
csvfile2=pd.read_csv('../../result-data/RowMerge_V16.csv',usecols=[1,2,3,4,5,6,7,8,9],names=["ROW","COL","NNZ","new_sparsity", "all_zero_vec_cnt", "CSR_storage", "CVSE_storage", "TCF_storage", "SR_BCSR_storage"])


sparsity = []
ROW = []
COL = []
NNZ = []
new_sparsity_8 = []
new_sparsity_16 = []
cnt8 = []
cnt16 = []
CSR_storage = []

DENSE_storage = []

CVSE_storage8 = []
TCF_storage8 = []
SR_BCSR_storage8 = []
CVSE_storage16 = []
TCF_storage16 = []
SR_BCSR_storage16 = []

matrix_num = len(csvfile1.ROW)

CVSE_storage8_norm = []
TCF_storage8_norm = []
SR_BCSR_storage8_norm = []
CVSE_storage16_norm = []
TCF_storage16_norm = []
SR_BCSR_storage16_norm = []

CSR_storage_norm1 = []
CVSE_storage8_norm1 = []
TCF_storage8_norm1 = []
SR_BCSR_storage8_norm1 = []
CVSE_storage16_norm1 = []
TCF_storage16_norm1 = []
SR_BCSR_storage16_norm1 = []
valid_num = 0
for i in range(matrix_num):
    if(int(csvfile1.NNZ[i]) != 0):
        valid_num += 1
        ROW.append(int(csvfile1.ROW[i]))
        COL.append(int(csvfile1.COL[i]))
        NNZ.append(int(csvfile1.NNZ[i]))
        CSR_storage.append(int(csvfile1.CSR_storage[i]))
        new_sparsity_8.append(float(csvfile1.new_sparsity[i]))
        new_sparsity_16.append(float(csvfile2.new_sparsity[i]))
        cnt8.append(int(csvfile1.all_zero_vec_cnt[i]))
        cnt16.append(int(csvfile2.all_zero_vec_cnt[i]))
        CVSE_storage8.append(int(csvfile1.CVSE_storage[i]))
        TCF_storage8.append(int(csvfile1.TCF_storage[i]))
        SR_BCSR_storage8.append(int(csvfile1.SR_BCSR_storage[i]))
        CVSE_storage16.append(int(csvfile2.CVSE_storage[i]))
        TCF_storage16.append(int(csvfile2.TCF_storage[i]))
        SR_BCSR_storage16.append(int(csvfile2.SR_BCSR_storage[i]))

print("File_Num = ", matrix_num, " Valid_Num = ", valid_num)

for i in range(valid_num):
    ratio = 1.0 - NNZ[i]/(ROW[i]*COL[i])
    sparsity.append(ratio)
    # DENSE_storage.append(ROW[i]*COL[i]*2)
    
    CVSE_storage8_norm.append(float(CVSE_storage8[i] / CSR_storage[i]))
    TCF_storage8_norm.append(float(TCF_storage8[i] / CSR_storage[i]))
    SR_BCSR_storage8_norm.append(float(SR_BCSR_storage8[i] / CSR_storage[i]))
    CVSE_storage16_norm.append(float(CVSE_storage16[i] / CSR_storage[i]))
    TCF_storage16_norm.append(float(TCF_storage16[i] / CSR_storage[i]))
    SR_BCSR_storage16_norm.append(float(SR_BCSR_storage16[i] / CSR_storage[i]))
    
    CSR_storage_norm1.append(float(CSR_storage[i] / (ROW[i]*COL[i]*2)))
    CVSE_storage8_norm1.append(float(CVSE_storage8[i] / (ROW[i]*COL[i]*2)))
    TCF_storage8_norm1.append(float(TCF_storage8[i] / (ROW[i]*COL[i]*2)))
    SR_BCSR_storage8_norm1.append(float(SR_BCSR_storage8[i] / (ROW[i]*COL[i]*2)))
    CVSE_storage16_norm1.append(float(CVSE_storage16[i] / (ROW[i]*COL[i]*2)))
    TCF_storage16_norm1.append(float(TCF_storage16[i] / (ROW[i]*COL[i]*2)))
    SR_BCSR_storage16_norm1.append(float(SR_BCSR_storage16[i] / (ROW[i]*COL[i]*2)))


CVSE_storage8_norm_average  = np.mean(CVSE_storage8_norm)
TCF_storage8_norm_average = np.mean(TCF_storage8_norm)
SR_BCSR_storage8_norm_average  = np.mean(SR_BCSR_storage8_norm)
CVSE_storage16_norm_average = np.mean(CVSE_storage16_norm)
TCF_storage16_norm_average  = np.mean(TCF_storage16_norm)
SR_BCSR_storage16_norm_average = np.mean(SR_BCSR_storage16_norm)

CVSE_storage8_norm1_average  = np.mean(CVSE_storage8_norm1)
CVSE_storage16_norm1_average = np.mean(CVSE_storage16_norm1)
TCF_storage8_norm1_average = np.mean(TCF_storage8_norm1)
TCF_storage16_norm1_average  = np.mean(TCF_storage16_norm1)
SR_BCSR_storage8_norm1_average  = np.mean(SR_BCSR_storage8_norm1)
SR_BCSR_storage16_norm1_average = np.mean(SR_BCSR_storage16_norm1)
print("CVSE_storage8_norm_average: ", CVSE_storage8_norm_average)
print("TCF_storage8_norm_average: ", TCF_storage8_norm_average)
# print("SR_BCSR_storage8_norm_average: ", SR_BCSR_storage8_norm_average)
print("CVSE_storage16_norm_average: ", CVSE_storage16_norm_average)
print("TCF_storage16_norm_average: ", TCF_storage16_norm_average)
# print("SR_BCSR_storage16_norm_average: ", SR_BCSR_storage16_norm_average)
print("CVSE_storage8_norm1_average: ", CVSE_storage8_norm1_average)
print("TCF_storage8_norm1_average: ", TCF_storage8_norm1_average)
# print("SR_BCSR_storage8_norm1_average: ", SR_BCSR_storage8_norm1_average)
print("CVSE_storage16_norm1_average: ", CVSE_storage16_norm1_average)
print("TCF_storage16_norm1_average: ", TCF_storage16_norm1_average)
# print("SR_BCSR_storage16_norm1_average: ", SR_BCSR_storage16_norm1_average)

csvfile3=pd.read_csv('../../result-data/RowMerge_suit_V8.csv' , usecols=[0,2,4,5,6,7], names=["name", "col", "CSR_storage", "CVSE_storage", "TCF_storage", "SR_BCSR_storage"])
csvfile4=pd.read_csv('../../result-data/RowMerge_suit_V16.csv', usecols=[0,2,4,5,6,7], names=["name", "col", "CSR_storage", "CVSE_storage", "TCF_storage", "SR_BCSR_storage"])
NAME_suit = []
COL_suit = []
CSR_storage_suit = []
DENSE_storage_suit = []
CVSE8_storage_suit  = []
CVSE16_storage_suit = []
TCF8_storage_suit  = []
TCF16_storage_suit = []
SRBCSR8_storage_suit = []
SRBCSR16_storage_suit = []

suit_num = len(csvfile3.col)
for i in range(suit_num):
    NAME_suit.append(csvfile3.name[i])
    COL_suit.append(int(csvfile3.col[i]))
    CSR_storage_suit.append(int(csvfile3.CSR_storage[i]))
    DENSE_storage_suit.append(COL_suit[i] * COL_suit[i] * 2)
    CVSE8_storage_suit.append(int(csvfile3.CVSE_storage[i]))
    CVSE16_storage_suit.append(int(csvfile4.CVSE_storage[i]))
    TCF8_storage_suit.append(int(csvfile3.TCF_storage[i]))
    TCF16_storage_suit.append(int(csvfile4.TCF_storage[i]))
    SRBCSR8_storage_suit.append(int(csvfile3.SR_BCSR_storage[i]))
    SRBCSR16_storage_suit.append(int(csvfile4.SR_BCSR_storage[i]))

suit_norm = CSR_storage_suit
CVSE8_suit_norm = []
CVSE16_suit_norm = []
TCF8_suit_norm = []
TCF16_suit_norm = []
SRBCSR8_suit_norm = []
SRBCSR16_suit_norm = []
for i in range(suit_num):
    CVSE8_suit_norm.append(float(CVSE8_storage_suit[i] / suit_norm[i]))
    CVSE16_suit_norm.append(float(CVSE16_storage_suit[i] / suit_norm[i]))
    TCF8_suit_norm.append(float(TCF8_storage_suit[i] / suit_norm[i]))
    TCF16_suit_norm.append(float(TCF16_storage_suit[i] / suit_norm[i]))
    SRBCSR8_suit_norm.append(float(SRBCSR8_storage_suit[i] / suit_norm[i]))
    SRBCSR16_suit_norm.append(float(SRBCSR16_storage_suit[i] / suit_norm[i]))
    
print("SuiteSparse Norm: new Format / CSR")
for i in range(suit_num):
    print(NAME_suit[i])
    print("CVSE8 : ", CVSE8_suit_norm[i],  "\nTCF8 : ", TCF8_suit_norm[i])
    print("CVSE16: ", CVSE16_suit_norm[i],  "\nTCF16: ", TCF16_suit_norm[i])
    
# print(CVSE8_suit_norm)
# print(CVSE16_suit_norm)
fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1,1]})
# fig, axs = plt.subplots(1, 1, figsize=(18, 8))
plt.subplot(2,1,1)
# plt.scatter(sparsity, CSR_storage_norm1,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='CSR / DENSE')
plt.scatter(sparsity, TCF_storage8_norm,s=50,c='#ffb4c8',marker='o',linewidth='0.0',label='ME-TCF-V8/CSR')
plt.scatter(sparsity, TCF_storage16_norm,s=50,c='#81d8cf',marker='o',linewidth='0.0',label='ME-TCF-V16/CSR')
# plt.scatter(sparsity, SR_BCSR_storage8_norm,s=50,c='#4eff9f',marker='p',linewidth='0.0',label='SR_BCSR-V8/CSR')
# plt.scatter(sparsity, SR_BCSR_storage16_norm,s=50,c='#eeff5b',marker='p',linewidth='0.0',label='SR_BCSR-V16/CSR')
plt.scatter(sparsity, CVSE_storage8_norm,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='CVSE-V8/CSR')
plt.scatter(sparsity, CVSE_storage16_norm,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='CVSE-V16/CSR')

# x = np.linspace(0.5, 1, 1000)
# y = [1 for i in range(1000)]
# plt.plot(x, y,c='black')
plt.legend(loc="upper left", borderaxespad=0,fontsize=23,ncol=2)
plt.ylim(0,5.5)
plt.xlim(0.5,1)
plt.xlabel("Sparsity",fontsize=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
# plt.tick_params(labelsize=24)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)
axs[0].set_title('Deep Learning Matrix Collection',fontsize=20)



plt.subplot(2,1,2)
x_label = [str((i+1)) for i in range(suit_num)]
# x_label = suit_names
# print(x_label)
x = np.arange(suit_num)
width = 0.11

# plt.bar(x - 5 * width, SRBCSR8_suit_norm , 2*width,color='#a6d8cf',edgecolor='black',linewidth=2,label='SR_BCSR-V8/DENSE')
# plt.bar(x - 3 * width, SRBCSR16_suit_norm, 2*width,color='#f3a1af',edgecolor='black',linewidth=2,label='SR_BCSR-V16/DENSE')
plt.bar(x - 3 * width, TCF8_suit_norm, 2*width,color='#ffb4c8',edgecolor='black',linewidth=2,label='ME-TCF-V8/CSR')
plt.bar(x - 1 * width, TCF16_suit_norm, 2*width,color='#81d8cf',edgecolor='black',linewidth=2,label='ME-TCF-V16/CSR')
plt.bar(x + 1 * width, CVSE8_suit_norm , 2*width,color='#ee6a5b',edgecolor='black',linewidth=2,label='CVSE-V8/CSR')
plt.bar(x + 3 * width, CVSE16_suit_norm, 2*width,color='#4ea59f',edgecolor='black',linewidth=2,label='CVSE-V16/CSR')
plt.ylim(0,4.2)
plt.yticks(np.arange(0, 4.2, 1))
plt.xlabel("Matrix ID",fontsize=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
axs[1].set_title('SuiteSparse Matrix Collection(Selected)',fontsize=20)
axs[1].set_xticks(x)
axs[1].set_xticklabels(x_label,rotation=0, fontsize = 20)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)
plt.tight_layout()


plt.subplots_adjust(left = 0.1, right = 0.98)
fig.text(0.01, 0.5, 'Normalized Storage Overhead (to CSR)', va='center', rotation='vertical', fontsize = 22)
fig.text(0.235, 0.43, '4.98', va='center', rotation=0, fontsize = 18)
fig.text(0.29,  0.43, '5.06', va='center', rotation=0, fontsize = 18)
fig.text(0.83,  0.43, '5.35', va='center', rotation=0, fontsize = 18)
# plt.tight_layout()
plt.savefig('exp1-2.pdf',dpi=2000)

