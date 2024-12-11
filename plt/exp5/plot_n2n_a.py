import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#sns.set_context(rc = {'patch.linewidth': 0.0})
order = ['Pytorch-fp16', 'vectorSparse', 'TAUS']
color = {'Pytorch-fp16': '#4ea59f', 'vectorSparse': '#81d8cf', 'TAUS': '#ffb4c8'}

n2n_a_data = pd.read_csv('n2n_a.csv')
n2n_b_data = pd.read_csv('n2n_b.csv')
n2n_e_data = pd.read_csv('n2n_e.csv')
n2n_f_data = pd.read_csv('n2n_f.csv')



fig, axs = plt.subplots(2, 2, figsize=(12, 10))

plt.subplot(2,2,1)
g = sns.barplot(data=n2n_a_data, x="S0.9,Seq_l=4096,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
axs[0, 0].set_title('S=0.9, H=4',fontsize=25)
axs[0, 0].set_ylabel("Latency(ms)",fontsize=25)
axs[0, 0].set_xlabel(" ",fontsize=1)
# axs[0, 0].set(ylim=(0, 25))
plt.ylim(0,30)
plt.yticks(range(0, 31, 5))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=25)
handles, labels = g.get_legend_handles_labels()
# [spine.set_linewidth(1.5) for spine in axs[0, 0].spines.values()]
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
    
plt.subplot(2,2,2)
g = sns.barplot(data=n2n_b_data, x="S0.9,Seq_l=4096,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
axs[0, 1].set_title('S=0.9, H=8',fontsize=25)
axs[0, 1].set_ylabel("Latency(ms)",fontsize=25)
axs[0, 1].set_xlabel(" ",fontsize=1)
# axs[0, 1].set(ylim=(0, 55))
plt.ylim(0,60)
plt.yticks(range(0, 61, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=25)

plt.subplot(2,2,3)
g = sns.barplot(data=n2n_e_data, x="S0.95,Seq_l=4096,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
axs[1, 0].set_title('S=0.95, H=4',fontsize=25)
axs[1, 0].set_ylabel("Latency(ms)",fontsize=25)
axs[1, 0].set_xlabel(" ",fontsize=45)
# axs[1, 0].set(ylim=(0, 25))
plt.ylim(0,30)
plt.yticks(range(0, 31, 5))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=25)

plt.subplot(2,2,4)
g = sns.barplot(data=n2n_f_data, x="S0.95,Seq_l=4096,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
axs[1, 1].set_title('S=0.95, H=8',fontsize=25)
axs[1, 1].set_ylabel("Latency(ms)",fontsize=25)
axs[1, 1].set_xlabel(" ",fontsize=45)
# axs[1, 1].set(ylim=(0, 55))
plt.ylim(0,60)
plt.yticks(range(0, 61, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=25)

sns.despine(offset=0, trim=False, top=False, right=False, left=False, bottom=False)

fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=25)
plt.tight_layout()
plt.savefig('exp5-0.pdf',dpi=300)

# plt.legend(loc="upper center", borderaxespad=0,fontsize=25,ncol=3)
