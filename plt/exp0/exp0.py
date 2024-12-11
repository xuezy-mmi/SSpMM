import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def list_mean(list_):
    length = len(list_)
    sum = 0.0
    for i in range(length):
        sum += list_[i]
    Lmean = sum / length
    return Lmean
vs1 = [12.26, 7.75, 3.43, 6.73]
vs2 = [10.88, 7.34, 3.12, 6.09]
vs3 = [7.30, 5.52, 2.42, 4.58]
vs4 = [2.44, 2.38, 1.27, 2.31]
sputnik1 = [2.92, 2.41, 4.82, 8.76]
sputnik2 = [2.54, 2.11, 3.67, 6.11]
sputnik3 = [1.43, 1.37, 1.95, 2.94]
sputnik4 = [0.59, 0.54, 0.70, 1.04]

x_line = ["Volta", "Turing", "Ampere", "Ada"]
peak_TC = [125, 108, 312, 330]
peak_CUDA = [31.4, 28.5, 78, 82.6]
percent_TC1 = [0.0,0.0,0.0,0.0]
percent_CUDA1 = [0.0,0.0,0.0,0.0]
percent_TC2 = [0.0,0.0,0.0,0.0]
percent_CUDA2 = [0.0,0.0,0.0,0.0]
percent_TC3 = [0.0,0.0,0.0,0.0]
percent_CUDA3 = [0.0,0.0,0.0,0.0]
percent_TC4 = [0.0,0.0,0.0,0.0]
percent_CUDA4 = [0.0,0.0,0.0,0.0]
for i in range(4):
    percent_TC1[i] = vs1[i] / peak_TC[i] * 100
    percent_CUDA1[i] = sputnik1[i] / peak_CUDA[i] * 100
    percent_TC2[i] = vs2[i] / peak_TC[i] * 100
    percent_CUDA2[i] = sputnik2[i] / peak_CUDA[i] * 100
    percent_TC3[i] = vs3[i] / peak_TC[i] * 100
    percent_CUDA3[i] = sputnik3[i] / peak_CUDA[i] * 100
    percent_TC4[i] = vs4[i] / peak_TC[i] * 100
    percent_CUDA4[i] = sputnik4[i] / peak_CUDA[i] * 100
print(percent_TC1)
print(percent_CUDA1)
print(percent_TC2)
print(percent_CUDA2)
print(percent_TC3)
print(percent_CUDA3)
print(percent_TC4)
print(percent_CUDA4)
# fig,ax = plt.subplots(figsize=(13, 7))
fig, ax = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1,1]})
x = np.arange(len(x_line))
width = 0.3

plt.subplot(2,2,1)
plt.bar(x - 0.5*width, sputnik1, width,color='#4ea59f',edgecolor='black',linewidth=2,label='Sputnik')
plt.bar(x + 0.5*width, vs1, width,color='#ee6a5b',edgecolor='black', linewidth=2,label='vectorSparse')
# for a,b in zip(x,sputnik1): ##label position
#     plt.text(a - 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# for a,b in zip(x,vs1): ##label position
#     plt.text(a + 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)

plt.ylabel(' ',fontsize=15)
ax[0,0].set_ylim(0, 13)
plt.yticks(range(0, 13, 4))
# plt.yticks(np.arange(0, 13, 2))
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(x_line,rotation=0, fontsize = 28)
plt.tick_params(labelsize=28)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper left", borderaxespad=0,fontsize=24,ncol=1)
ax1 = ax[0,0].twinx()
# ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x-0.5*width, percent_CUDA1, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'green', label = 'Sputnik Peformance / CUDA Core peak')
plt.plot(x+0.5*width, percent_TC1, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'red', label = 'vectorSparse Peformance / TC peak')
ax1.set_ylabel(' ', fontsize=0)
plt.yticks(range(0, 13, 4))
plt.tick_params(labelsize=28)
# plt.legend(loc="upper right", borderaxespad=0,fontsize=24,ncol=1)
plt.subplot(2,2,2)
plt.bar(x - 0.5*width, sputnik2, width,color='#4ea59f',edgecolor='black',linewidth=2,label='Sputnik')
plt.bar(x + 0.5*width, vs2, width,color='#ee6a5b',edgecolor='black', linewidth=2,label='vectorSparse')
# for a,b in zip(x,sputnik2): ##label position
#     plt.text(a - 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# for a,b in zip(x,vs2): ##label position
#     plt.text(a + 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)

plt.ylabel(' ',fontsize=0)
ax[0,1].set_ylim(0, 13)
plt.yticks(range(0, 13, 4))
# plt.yticks(np.arange(0, 13, 2))
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(x_line,rotation=0, fontsize = 28)
plt.tick_params(labelsize=28)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper left", borderaxespad=0,fontsize=24,ncol=1)
ax1 = ax[0,1].twinx()
# ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x-0.5*width, percent_CUDA2, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'green', label = 'Sputnik Peformance / CUDA Core peak')
plt.plot(x+0.5*width, percent_TC2, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'red', label = 'vectorSparse Peformance / TC peak')
ax1.set_ylabel(' ', fontsize=28)
plt.yticks(range(0, 10, 3))
plt.tick_params(labelsize=28)

plt.subplot(2,2,3)
plt.bar(x - 0.5*width, sputnik3, width,color='#4ea59f',edgecolor='black',linewidth=2,label='Sputnik')
plt.bar(x + 0.5*width, vs3, width,color='#ee6a5b',edgecolor='black', linewidth=2,label='vectorSparse')
# for a,b in zip(x,sputnik3): ##label position
#     plt.text(a - 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# for a,b in zip(x,vs3): ##label position
#     plt.text(a + 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
plt.xlabel(' ',fontsize=80)
plt.ylabel(' ',fontsize=15)
plt.yticks(range(0, 10, 3))
# plt.yticks(np.arange(0, 13, 2))
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(x_line,rotation=0, fontsize = 28)
plt.tick_params(labelsize=28)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper left", borderaxespad=0,fontsize=24,ncol=1)
ax1 = ax[1,0].twinx()
# ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x-0.5*width, percent_CUDA3, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'green', label = 'Sputnik Peformance / CUDA Core peak')
plt.plot(x+0.5*width, percent_TC3, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'red', label = 'vectorSparse Peformance / TC peak')
ax1.set_ylabel(' ', fontsize=0)
plt.yticks(range(0, 7, 2))
plt.tick_params(labelsize=28)

g = plt.subplot(2,2,4)
plt.bar(x - 0.5*width, sputnik4, width,color='#4ea59f',edgecolor='black',linewidth=2,label='Sputnik')
plt.bar(x + 0.5*width, vs4, width,color='#ee6a5b',edgecolor='black', linewidth=2,label='vectorSparse')
# for a,b in zip(x,sputnik4): ##label position
#     plt.text(a - 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
# for a,b in zip(x,vs4): ##label position
#     plt.text(a + 0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=22)
plt.xlabel(' ',fontsize=80)
plt.ylabel(' ',fontsize=0)
plt.yticks(range(0, 5, 1))
# plt.yticks(np.arange(0, 13, 2))
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(x_line,rotation=0, fontsize = 28)
plt.tick_params(labelsize=28)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper left", borderaxespad=0,fontsize=24,ncol=1)
ax1 = ax[1,1].twinx()
# ax2.axhline(y=1, color='black', linestyle='--')
plt.plot(x-0.5*width, percent_CUDA4, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'green', label = 'Sputnik Peformance / CUDA Core peak')
plt.plot(x+0.5*width, percent_TC4, marker='o', linestyle='-', linewidth=3,markersize = 10,color = 'red', label = 'vectorSparse Peformance / TC peak')
ax1.set_ylabel(' ', fontsize=28)
ax1.set_ylim(0, 3)
plt.tick_params(labelsize=28)

handles1, labels1 = g.get_legend_handles_labels()
handles2, labels2 = ax1.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=24)
fig.text(0.001, 0.6, 'Performance (TFLOPS)', va='center', rotation='vertical', fontsize = 28)
fig.text(0.97, 0.6, 'Percentage (%)', va='center', rotation='vertical', fontsize = 28)
fig.text(0.28, 0.93, 'sparsity=0.5', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.75, 0.93, 'sparsity=0.7', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.28, 0.51, 'sparsity=0.9', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.75, 0.51, 'sparsity=0.98', va='center', rotation='horizontal', fontsize = 28)
fig.tight_layout()
plt.savefig('exp0-0.pdf',dpi=300)