import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



csv8  = pd.read_csv('../../result-data/turing_SSpMM_v8.csv' ,usecols=[0, 4],names=["nnz", "perf"])
csv16 = pd.read_csv('../../result-data/turing_SSpMM_v16.csv',usecols=[0, 4],names=["nnz", "perf"])

suitcsv8  = pd.read_csv('../../result-data/turing_suit_SSpMM_v8.csv' ,usecols=[4],names=["perf"])
suitcsv16 = pd.read_csv('../../result-data/turing_suit_SSpMM_v16.csv',usecols=[4],names=["perf"])

sp = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
sparsity0 = []
sparsity1 = []
perf0 = []
perf1 = []
perf2 = []
perf3 = []
type0 = []
type1 = []
filenum = 2688
suit_num = len(suitcsv16.perf)
valid_num = 0
print("Total dlmc data num: ", filenum)
for i in range (filenum):
    if(int(csv16.nnz[i])!=0):
        valid_num += 1
        perf0.append(float(csv8.perf[i]))
        type0.append('SSpMM-V8')
        perf1.append(float(csv16.perf[i]))
        type1.append('SSpMM-V16')
print("Valid dlmc data num: ", valid_num)        
for i in range(suit_num):
    perf2.append(float(suitcsv8.perf[i]))
    perf3.append(float(suitcsv16.perf[i]))

for i in range(4):
    for j in range(7):
        xlabel = sp[j]
        for k in range(96):
            NO_file = i*7*96+j*96+k
            if(int(csv16.nnz[NO_file])!=0):
                sparsity0.append(xlabel)
                sparsity1.append(xlabel)

df_sspmm8 = pd.DataFrame({
    'sparsity': sparsity0,
    'perf': perf0,
    'lib': type0
})
df_sspmm16 = pd.DataFrame({
    'sparsity': sparsity1,
    'perf': perf1,
    'lib': type1
})

v8_mean  = df_sspmm8.groupby('sparsity')['perf'].mean()
v16_mean = df_sspmm16.groupby('sparsity')['perf'].mean()
trade_off = [0.0 for i in range(7)]
trade_off_suit = [0.0 for i in range(suit_num)]

for i in range(7):
    print("Mean_Perf: ", sp[i], "\t", v8_mean[i], v16_mean[i])
    trade_off[i] = v8_mean[i] / v16_mean[i]
    
for i in range(7):
    print(sp[i], "  trade-off")
    print("ampere  8-16: ", trade_off[i])

for i in range(suit_num):
    trade_off_suit[i] = perf2[i] / perf3[i]
    
    
x = np.arange(len(sp))  # the label locations
# y = np.arange(len(men_means))
width = 0.35  # the width of the bars

fig,ax = plt.subplots(1, 2, figsize=(30, 5.2), gridspec_kw={'width_ratios': [1,2]})
# fig, axs = plt.subplots(2, 2, figsize=(20, 10))
plt.subplot(1,2,1)
plt.bar(x - 0.5*width, v8_mean , width,color='#ee6a5b',edgecolor='black',linewidth=2,label='SSpMM-V8')
plt.bar(x + 0.5*width, v16_mean, width,color='#4ea59f',edgecolor='black', linewidth=2,label='SSpMM-V16')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=24)
plt.yticks(range(0, 7001, 1000))
# plt.title('Scores by group and gender',fontsize=28)
ax[0].set_xticks(x)
ax[0].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[0].set_title('Deep Learning Matrix Collection',fontsize=24)
plt.xlabel('Sparsity',fontsize=25)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=2)
for a,b in zip(x,v8_mean): ##label position
    plt.text(a - 0.5*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
for a,b in zip(x,v16_mean): ##label position
    plt.text(a + 0.5*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# plt.legend([], [], frameon=False)

ax2 = ax[0].twinx()
# base = [1 for i in range(7)]
# plt.plot(x, base, color = 'gray')
ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x, trade_off, marker='o', linestyle='-', markersize = 12, linewidth = 2,color = 'blue', label = 'SSpMM-SVC8 / SSpMM-SVC16')
ax2.set_ylabel('', fontsize=0)
ax2.set_ylim(0, 1.6)
plt.tick_params(labelsize=22)

# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

g = plt.subplot(1,2,2)

x = np.arange(suit_num)
x_label = [str(i+1) for i in range(suit_num)]
plt.bar(x - 0.5*width, perf2 , width,color='#ee6a5b',edgecolor='black',linewidth=2,label='SSpMM-V8')
plt.bar(x + 0.5*width, perf3, width,color='#4ea59f',edgecolor='black', linewidth=2,label='SSpMM-V16')
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('',fontsize=0)
plt.yticks(range(0, 6001, 1000))
# plt.title('Scores by group and gender',fontsize=28)
ax[1].set_xticks(x)
ax[1].set_xticklabels(x_label,rotation=0, fontsize = 22)
ax[1].set_title('SuiteSparse Matrix Collection',fontsize=24)
plt.xlabel('Matrix ID',fontsize=25)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')

for a,b in zip(x,perf2): ##label position
    plt.text(a - 0.5*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
for a,b in zip(x,perf3): ##label position
    plt.text(a + 0.5*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# plt.legend([], [], frameon=False)

ax2 = ax[1].twinx()
ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x, trade_off_suit, marker='o', linestyle='-', markersize = 12, linewidth = 2, color = 'blue', label = 'Performance Ratio of SSpMM-V8 and SSpMM-V16')
ax2.set_ylabel(' ', fontsize=22)
ax2.set_ylim(0, 1.6)
plt.tick_params(labelsize=22)
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=2)



# handles, labels = g.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=30)

fig.text(0.001, 0.5, 'Performance (GFLOPS)', va='center', rotation='vertical', fontsize = 28)
fig.text(0.99, 0.5, 'V8 / V16 (ratio)', va='center', rotation='vertical', fontsize = 26)
fig.tight_layout()
plt.savefig('exp3.pdf',dpi=300)
# plt.show()
