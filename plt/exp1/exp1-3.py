import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

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
CVSE_storage8 = []
TCF_storage8 = []
SR_BCSR_storage8 = []
CVSE_storage16 = []
TCF_storage16 = []
SR_BCSR_storage16 = []

matrix_num = len(csvfile1.ROW)
for i in range(matrix_num):
    ROW.append(int(csvfile1.ROW[i]))
    COL.append(int(csvfile1.COL[i]))
    NNZ.append(int(csvfile1.NNZ[i]))
    CSR_storage.append(int(csvfile1.NNZ[i]))
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

for i in range(matrix_num):
    ratio = 1.0 - NNZ[i]/(ROW[i]*COL[i])
    sparsity.append(ratio)
    
print(matrix_num)


# fig=plt.figure(figsize=(40,15))
fig, axs = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1,1]})
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()

plt.subplot(2,1,1)

# 4ea59f deep green 
# ee6a5b deep red
plt.scatter(sparsity, cnt8,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='number of zero vectors (V=8)')
plt.scatter(sparsity, cnt16,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='number of zero vectors (V=16)')

plt.legend(loc="upper left",fontsize=28,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=10)
plt.ylim(0,140000)
plt.yticks(range(0, 140001, 20000))
plt.xlim(0.5,1)
plt.xlabel("Sparsity",fontsize=24)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)

plt.subplot(2,1,2)
# plt.plot([0, 1], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
# 81d8cf light green
# ffb4c8 light red
plt.scatter(sparsity, cnt8,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='number of zero vectors (V=8)')
plt.scatter(sparsity, cnt16,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='number of zero vectors (V=16)')
# plt.legend(loc="upper left",fontsize=28,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=20)
plt.ylim(0,140000)
plt.yticks(range(0, 140001, 20000))
plt.xlim(0.5,1)
plt.xlabel("Sparsity",fontsize=24)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)
plt.tight_layout()

fig.subplots_adjust(left = 0.1, right = 0.98)
# fig.subplots_adjust(left = 0.2, right = 0.95, wspace = 0.1, hspace= 0.1)
# plt.tight_layout()
fig.text(0.01, 0.5, 'Number of Zero-Vectors', va='center', rotation='vertical', fontsize = 24)

# plt.tight_layout()

plt.savefig('exp1_1.pdf',dpi=2000)

# plt.show()
