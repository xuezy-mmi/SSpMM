import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

def Fmean(array):
    len_a = len(array)
    sum_a = sum(array)
    return float(sum_a / len_a)

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
# dense_dlmc_8X = [0.0 for i in range()]
CSR_storage = []
CVSE_storage8 = []
TCF_storage8 = []
SR_BCSR_storage8 = []
CVSE_storage16 = []
TCF_storage16 = []
SR_BCSR_storage16 = []
matrix_num = len(csvfile1.ROW)
valid_num = 0
for i in range(matrix_num):
    if(int(csvfile1.NNZ[i]) != 0):
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
        valid_num += 1

dense_dlmc_8X = []
dense_dlmc_16X = []
for i in range(valid_num):
    ratio = 1.0 - NNZ[i]/(ROW[i]*COL[i])
    dense_dlmc_8X.append((1-new_sparsity_8[i])/(1-ratio))
    dense_dlmc_16X.append((1-new_sparsity_16[i])/(1-ratio))
    sparsity.append(ratio)
print("Total Num = ", matrix_num, " Valid Num = ", valid_num)
print("DLMC DenseX:\n", Fmean(dense_dlmc_8X))
print(Fmean(dense_dlmc_16X))

csvfile3=pd.read_csv('../../result-data/RowMerge_suit_V8.csv' ,usecols=[0, 1, 2, 3, 8], names=["suit_name", "NNZ", "COL", "new_sparsity", "dense_x"])
csvfile4=pd.read_csv('../../result-data/RowMerge_suit_V16.csv',usecols=[0, 1, 2, 3, 8], names=["suit_name", "NNZ", "COL", "new_sparsity", "dense_x"])

new_suit_sparsity_8 = []
new_suit_sparsity_16 = []
suit_names = []
suit_NNZ = []
suit_COL = []
dense_8X  = []
dense_16X = []
suit_density = []
suit_num = len(csvfile3.new_sparsity)
for i in range(suit_num):
    suit_names.append(csvfile3.suit_name[i])
    suit_NNZ.append(csvfile3.NNZ[i])
    suit_COL.append(csvfile3.COL[i])
    new_suit_sparsity_8.append(float(csvfile3.new_sparsity[i]))
    new_suit_sparsity_16.append(float(csvfile4.new_sparsity[i]))
    dense_8X.append(float(csvfile3.dense_x[i]))
    dense_16X.append(float(csvfile4.dense_x[i]))

dense_8X_mean = Fmean(dense_8X)
dense_16X_mean = Fmean(dense_16X)

# fig=plt.figure(figsize=(40,15))
fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1,1]})
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()

plt.subplot(2,1,1)

# 4ea59f deep green 
# ee6a5b deep red
plt.scatter(sparsity, new_sparsity_8,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='V=8')
plt.scatter(sparsity, new_sparsity_16,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='V=16')
x = np.linspace(0.5, 1, 1000)
y = x
y8 = x*0 + 7/8
y16 = x*0 + 15/16
plt.plot(x, y,c='black')
# plt.plot(x, y8,c='#ee6a5b',linestyle='--')
# plt.plot(x, y16,c='#4ea59f',linestyle='--')
axs[0].axhline(y=7/8, color='#ee6a5b', linestyle='--')
axs[0].axhline(y=15/16, color='#4ea59f', linestyle='--')

plt.legend(loc="upper left",fontsize=20,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=10)
plt.ylim(0.4,1)
# plt.yticks(range(0, 1, 20000))
plt.xlim(0.5,1)
plt.xlabel("Sparsity of Matrix",fontsize=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
axs[0].set_title('Deep Learning Matrix Collection',fontsize=20)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)

plt.subplot(2,1,2)
x_label = [str(i+1) for i in range(suit_num)]
x = np.arange(suit_num)
width = 0.4
# 81d8cf light green
# ffb4c8 light red
plt.bar(x        , new_suit_sparsity_8 , 1*width,color='#ee6a5b',edgecolor='black',linewidth=2,label='V=8')
axs[1].axhline(y=7/8, color='#ee6a5b', linestyle='--')
plt.bar(x + width, new_suit_sparsity_16, 1*width,color='#4ea59f',edgecolor='black',linewidth=2,label='V=16')
axs[1].axhline(y=15/16, color='#4ea59f', linestyle='--')

for i in range(suit_num):
    plt.text(x[i] ,0,'%.1fX' %dense_8X[i],ha = 'center',va = 'bottom',rotation =90,fontsize=15)
    plt.text(x[i] + width ,0,'%.1fX' %dense_16X[i],ha = 'center',va = 'bottom',rotation =90,fontsize=15)
# for a,b in zip(x,dense_8X): ##third bar
    # plt.text(a+0*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
# for a,b in zip(x,dense_16X): ##forth bar
    # plt.text(a+1*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)

print("SuiteSparse DenseX:\n",Fmean(dense_8X))
print(Fmean(dense_16X))
plt.ylim(0,1)
plt.yticks(np.arange(0, 1.2, 0.2))

plt.xlabel("Matrix ID",fontsize=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
axs[1].set_title('SuiteSparse Matrix Collection(Selected)',fontsize=20)
axs[1].set_xticks(x)
# axs[1].set_yticks(np.arange(0, 1.2, 0.2))
axs[1].set_xticklabels(x_label,rotation=0, fontsize = 20)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)
plt.tight_layout()

fig.subplots_adjust(left = 0.1, right = 0.98)
# fig.subplots_adjust(left = 0.2, right = 0.95, wspace = 0.1, hspace= 0.1)
# plt.tight_layout()
fig.text(0.01, 0.5, 'New Sparsity in CVSE Format', va='center', rotation='vertical', fontsize = 24)

fig.text(0.058, 0.87, '7/8', va='center', rotation='horizontal', fontsize = 20, color='#ee6a5b')
fig.text(0.03, 0.905, '15/16', va='center', rotation='horizontal', fontsize = 20, color='#4ea59f')
# plt.tight_layout()

plt.savefig('exp1-1.pdf',dpi=2000)

# plt.show()
