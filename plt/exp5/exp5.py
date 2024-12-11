import re
import six
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#sns.set_context(rc = {'patch.linewidth': 0.0})
order = ['Pytorch-fp16', 'vectorSparse', 'RoSpMM']
color = {'Pytorch-fp16': '#81d8cf', 'vectorSparse': '#4ea59f', 'RoSpMM': '#ee6a5b'}

n2n_a_data = pd.read_csv('n2n_a.csv')
n2n_b_data = pd.read_csv('n2n_b.csv')
n2n_c_data = pd.read_csv('n2n_c.csv')
n2n_d_data = pd.read_csv('n2n_d.csv')
n2n_e_data = pd.read_csv('n2n_e.csv')
n2n_f_data = pd.read_csv('n2n_f.csv')
n2n_g_data = pd.read_csv('n2n_g.csv')
n2n_h_data = pd.read_csv('n2n_h.csv')

x_label = ["BS=2", "BS=8"]

fig, axs = plt.subplots(2, 4, figsize=(15, 9/16*15))

plt.subplot(2,4,1)
g = sns.barplot(data=n2n_a_data, x="S0.9,Seq_l=4096,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[0, 0].set_title('S=0.9, H=4, L=4096',fontsize=24)
axs[0, 0].set_title(' ',fontsize=30)
axs[0, 0].set_ylabel(" ",fontsize=24)
axs[0, 0].set_xlabel("",fontsize=0)
# axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,30)
plt.yticks(range(0, 31, 5))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)
handles, labels = g.get_legend_handles_labels()

    
plt.subplot(2,4,2)
g = sns.barplot(data=n2n_b_data, x="S0.9,Seq_l=4096,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[0, 1].set_title('S=0.9, H=8, L=4096',fontsize=24)
axs[0, 1].set_title(' ',fontsize=30)
axs[0, 1].set_ylabel("",fontsize=0)
axs[0, 1].set_xlabel("",fontsize=0)
axs[0, 1].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,60)
plt.yticks(range(0, 61, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,3)
g = sns.barplot(data=n2n_c_data, x="S0.9,Seq_l=8192,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[0, 2].set_title('S=0.9, H=4, L=8192',fontsize=24)
axs[0, 2].set_title(' ',fontsize=30)
axs[0, 2].set_ylabel("",fontsize=0)
axs[0, 2].set_xlabel("",fontsize=0)
axs[0, 2].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,80)
plt.yticks(range(0, 81, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,4)
g = sns.barplot(data=n2n_d_data, x="S0.9,Seq_l=8192,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[0, 3].set_title('S=0.9, H=8, L=8192',fontsize=24)
axs[0, 3].set_title(' ',fontsize=30)
axs[0, 3].set_ylabel("",fontsize=0)
axs[0, 3].set_xlabel("",fontsize=0)
axs[0, 3].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,160)
plt.yticks(range(0, 161, 20))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,5)
g = sns.barplot(data=n2n_e_data, x="S0.95,Seq_l=4096,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[1, 0].set_title('S=0.95, H=4, L=4096',fontsize=24)
axs[1, 0].set_title(' ',fontsize=20)
axs[1, 0].set_ylabel(" ",fontsize=24)
axs[1, 0].set_xlabel(" ",fontsize=45)
axs[1, 0].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,30)
plt.yticks(range(0, 31, 5))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,6)
g = sns.barplot(data=n2n_f_data, x="S0.95,Seq_l=4096,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[1, 1].set_title('S=0.95, H=8, L=4096',fontsize=24)
axs[1, 1].set_title(' ',fontsize=20)
axs[1, 1].set_ylabel("",fontsize=0)
axs[1, 1].set_xlabel(" ",fontsize=45)
axs[1, 1].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,60)
plt.yticks(range(0, 61, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,7)
g = sns.barplot(data=n2n_g_data, x="S0.95,Seq_l=8192,num_h=4", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[1, 2].set_title('S=0.95, H=4, L=8192',fontsize=24)
axs[1, 2].set_title(' ',fontsize=20)
axs[1, 2].set_ylabel("",fontsize=0)
axs[1, 2].set_xlabel(" ",fontsize=45)
axs[1, 2].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,60)
plt.yticks(range(0, 61, 10))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)


plt.subplot(2,4,8)
g = sns.barplot(data=n2n_h_data, x="S0.95,Seq_l=8192,num_h=8", y="Latency(ms)", hue="algs", palette=color,edgecolor='black', linewidth=2,hue_order=order, capsize=.1, errwidth=1)
# axs[1, 3].set_title('S=0.95, H=8, L=8192',fontsize=24)
axs[1, 3].set_title(' ',fontsize=20)
axs[1, 3].set_ylabel("",fontsize=0)
axs[1, 3].set_xlabel(" ",fontsize=45)
axs[1, 3].set_xticklabels(x_label,rotation=0, fontsize = 24)
plt.ylim(0,120)
plt.yticks(range(0, 121, 20))
plt.legend([], [], frameon=False)
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.tick_params(labelsize=24)

sns.despine(offset=0, trim=False, top=False, right=False, left=False, bottom=False)
labels[2] = "SSpMM"
fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=26)
fig.text(0.001, 0.5, 'Latency (ms)', va='center', rotation='vertical', fontsize = 28)
fig.text(0.42, 0.97, 'Sparsity = 0.9', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.42, 0.52, 'Sparsity = 0.95', va='center', rotation='horizontal', fontsize = 28)

fig.text(0.07, 0.88, 'H=4\nL=4096', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.07, 0.43, 'H=4\nL=4096', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.315, 0.88, 'H=8\nL=4096', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.315, 0.43, 'H=8\nL=4096', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.56, 0.88, 'H=8\nL=8192', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.56, 0.43, 'H=8\nL=8192', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.805, 0.88, 'H=8\nL=8192', va='center', rotation='horizontal', fontsize = 28)
fig.text(0.805, 0.43, 'H=8\nL=8192', va='center', rotation='horizontal', fontsize = 28)

fig.text(0.66, 0.65, 'OOM', va='center', rotation='vertical', fontsize = 28)
fig.text(0.905, 0.65, 'OOM', va='center', rotation='vertical', fontsize = 28)
fig.text(0.66, 0.2, 'OOM', va='center', rotation='vertical', fontsize = 28)
fig.text(0.905, 0.2, 'OOM', va='center', rotation='vertical', fontsize = 28)

plt.tight_layout()
plt.savefig('exp5.pdf',dpi=300)

# plt.legend(loc="upper center", borderaxespad=0,fontsize=25,ncol=3)
