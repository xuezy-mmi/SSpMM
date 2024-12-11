import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib import gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import readfile

Pcusparse1 = "../../data/volta_cusparse.csv"
Pcublas1 = "../../data/volta_cublas.csv"
Pvectorsparse1 = "../../data/volta_vectorsparse.csv"
Psputnik1 = "../../data/volta_sputnik.csv"
Prospmm81 = "../../data/volta_rospmm_8_v0.csv"
Prospmm161 = "../../data/volta_rospmm_16_v0.csv"
Pcusparse2 = "../../data/turing_cusparse.csv"
Pcublas2 = "../../data/turing_cublas.csv"
Pvectorsparse2 = "../../data/turing_vectorsparse.csv"
Psputnik2 = "../../data/turing_sputnik.csv"
Prospmm82 = "../../data/turing_rospmm_8_v0.csv"
Prospmm162 = "../../data/turing_rospmm_16_v0.csv"
Pcusparse3 = "../../data/ampere_cusparse.csv"###########
Pcublas3 = "../../data/ampere_cublas.csv"###########
Pvectorsparse3 = "../../data/ampere_vectorsparse.csv"
Psputnik3 = "../../data/ampere_sputnik.csv"
Prospmm83 = "../../data/ampere_rospmm_8_v0.csv"
Prospmm163 = "../../data/ampere_rospmm_16_v0.csv"
Pcusparse4 = "../../data/ada_cusparse.csv"############
Pcublas4 = "../../data/ada_cublas.csv"##########
Pvectorsparse4 = "../../data/ada_vectorsparse.csv"
Psputnik4 = "../../data/ada_sputnik.csv"
Prospmm84 = "../../data/ada_rospmm_8_v0.csv"
Prospmm164 = "../../data/ada_rospmm_16_v0.csv"
# Pmagicube = ""
csvfile0 = pd.read_csv('../../data/volta_cusparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile1 = pd.read_csv('../../data/volta_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile2 = pd.read_csv('../../data/volta_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile3 = pd.read_csv('../../data/volta_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile4 = pd.read_csv('../../data/volta_vectorsparse_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile5 = pd.read_csv('../../data/volta_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile6 = pd.read_csv('../../data/volta_rospmm_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile7 = pd.read_csv('../../data/turing_cusparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile8 = pd.read_csv('../../data/turing_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile9 = pd.read_csv('../../data/turing_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile10 = pd.read_csv('../../data/turing_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile11 = pd.read_csv('../../data/turing_vectorsparse_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile12 = pd.read_csv('../../data/turing_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile13 = pd.read_csv('../../data/turing_rospmm_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile14 = pd.read_csv('../../data/ampere_cusparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile15 = pd.read_csv('../../data/ampere_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile16 = pd.read_csv('../../data/ampere_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile17 = pd.read_csv('../../data/ampere_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile18 = pd.read_csv('../../data/ampere_vectorsparse_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile19 = pd.read_csv('../../data/ampere_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile20 = pd.read_csv('../../data/ampere_rospmm_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile21 = pd.read_csv('../../data/ada_cusparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile22 = pd.read_csv('../../data/ada_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile23 = pd.read_csv('../../data/ada_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile24 = pd.read_csv('../../data/ada_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile25 = pd.read_csv('../../data/ada_vectorsparse_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile26 = pd.read_csv('../../data/ada_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile27 = pd.read_csv('../../data/ada_rospmm_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

csvfile28 = pd.read_csv('../../data/magicube_rowmerge_L8R8.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile29 = pd.read_csv('../../data/magicube_rowmerge_L16R8.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile30 = pd.read_csv('../../data/magicube_rowmerge_L16R16.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

# total_mean, dis_mean = readfile.read_perf('../../data/magicube_rowmerge_L8R8.csv')
# total_mean, dis_mean = readfile.read_perf('../../data/magicube_rowmerge_L8R8.csv')
# total_mean, dis_mean = readfile.read_perf('../../data/magicube_rowmerge_L8R8.csv')

filenum = len(csvfile1.nnz)
valid_file = 0
print(filenum)

perf_sputnik_1 = [[] for i in range(7)]
perf_sputnik_2 = [[] for i in range(7)]
perf_sputnik_3 = [[] for i in range(7)]
perf_sputnik_4 = [[] for i in range(7)]
perf_sputnik_11 = [[] for i in range(7)]
perf_sputnik_22 = [[] for i in range(7)]
perf_sputnik_33 = [[] for i in range(7)]
perf_sputnik_44 = [[] for i in range(7)]
perf_cusparse_1 = [[] for i in range(7)]
perf_cusparse_2 = [[] for i in range(7)]
perf_cusparse_3 = [[] for i in range(7)]
perf_cusparse_4 = [[] for i in range(7)]
perf_vectorsparse_1 = [[] for i in range(7)]
perf_vectorsparse_2 = [[] for i in range(7)]
perf_vectorsparse_3 = [[] for i in range(7)]
perf_vectorsparse_4 = [[] for i in range(7)]
perf_vectorsparse8_1 = [[] for i in range(7)]
perf_vectorsparse8_2 = [[] for i in range(7)]
perf_vectorsparse8_3 = [[] for i in range(7)]
perf_vectorsparse8_4 = [[] for i in range(7)]
perf_vectorsparse16_1 = [[] for i in range(7)]
perf_vectorsparse16_2 = [[] for i in range(7)]
perf_vectorsparse16_3 = [[] for i in range(7)]
perf_vectorsparse16_4 = [[] for i in range(7)]
perf_rospmm8_1 = [[] for i in range(7)]
perf_rospmm8_2 = [[] for i in range(7)]
perf_rospmm8_3 = [[] for i in range(7)]
perf_rospmm8_4 = [[] for i in range(7)]
perf_rospmm16_1 = [[] for i in range(7)]
perf_rospmm16_2 = [[] for i in range(7)]
perf_rospmm16_3 = [[] for i in range(7)]
perf_rospmm16_4 = [[] for i in range(7)]

perf_magicube_rm_88 = [[] for i in range(7)]
perf_magicube_rm_168 = [[] for i in range(7)]
perf_magicube_rm_1616 = [[] for i in range(7)]

for i in range(4):
    for j in range(7):
        # xlabel = sp[j]
        for k in range(97):
            NO_file = i*7*97+j*97+k
            if(int(csvfile1.nnz[NO_file])!=0):
                if(k != 96):
                    perf_sputnik_1[j].append(float(csvfile1.perf[NO_file]))
                    perf_sputnik_2[j].append(float(csvfile8.perf[NO_file]))
                    perf_sputnik_3[j].append(float(csvfile15.perf[NO_file]))
                    perf_sputnik_4[j].append(float(csvfile22.perf[NO_file]))
                    valid_file += 1

def merge_array(array):
    len_a = len(array)
    new_array = []
    for i in range(len_a):
        new_array += array[i]
    return new_array
def Fmean(array):
    len_a = len(array)
    sum_a = sum(array)
    return float(sum_a / len_a)
def Imean(array):
    len_a = len(array)
    sum_a = sum(array)
    return int(sum_a / len_a)

mean_sputnik_1 = [[] for i in range(7)]
mean_sputnik_2 = [[] for i in range(7)]
mean_sputnik_3 = [[] for i in range(7)]
mean_sputnik_4 = [[] for i in range(7)]
for i in range(7):
    mean_sputnik_1[i] = Fmean(perf_sputnik_1[i])
    mean_sputnik_2[i] = Fmean(perf_sputnik_2[i])
    mean_sputnik_3[i] = Fmean(perf_sputnik_3[i])
    mean_sputnik_4[i] = Fmean(perf_sputnik_4[i])
    

print("baseline sputnik on Volta  : ", mean_sputnik_1)
print("baseline sputnik on Turing : ", mean_sputnik_2)
print("baseline sputnik on Ampere : ", mean_sputnik_3)
print("baseline sputnik on Ada    : ", mean_sputnik_4)

test_num = 0
for i in range(4):
    for j in range(7):
        # xlabel = sp[j]
        for k in range(97):
            NO_file = i*7*97+j*97+k
            if(int(csvfile1.nnz[NO_file])!=0):
                if(k != 96):
                    perf_sputnik_11[j].append(float(csvfile1.perf[NO_file]/mean_sputnik_1[j]))
                    perf_sputnik_22[j].append(float(csvfile8.perf[NO_file]/mean_sputnik_2[j]))
                    perf_sputnik_33[j].append(float(csvfile15.perf[NO_file]/mean_sputnik_3[j]))
                    perf_sputnik_44[j].append(float(csvfile22.perf[NO_file]/mean_sputnik_4[j]))
                    perf_cusparse_1[j].append(float(csvfile0.perf[NO_file]/mean_sputnik_1[j]))
                    perf_vectorsparse_1[j].append(float(csvfile2.perf[NO_file]/mean_sputnik_1[j]))
                    perf_vectorsparse8_1[j].append(float(csvfile3.perf[NO_file]/mean_sputnik_1[j]))
                    perf_vectorsparse16_1[j].append(float(csvfile4.perf[NO_file]/mean_sputnik_1[j]))
                    perf_cusparse_2[j].append(float(csvfile7.perf[NO_file]/mean_sputnik_2[j]))
                    perf_vectorsparse_2[j].append(float(csvfile9.perf[NO_file]/mean_sputnik_2[j]))
                    perf_vectorsparse8_2[j].append(float(csvfile10.perf[NO_file]/mean_sputnik_2[j]))
                    perf_vectorsparse16_2[j].append(float(csvfile11.perf[NO_file]/mean_sputnik_2[j]))
                    perf_rospmm8_2[j].append(float(csvfile12.perf[NO_file]/mean_sputnik_2[j]))
                    perf_rospmm16_2[j].append(float(csvfile13.perf[NO_file]/mean_sputnik_2[j]))
                    # perf_cusparse_3[j].append(float(csvfile14.perf[NO_file]/mean_sputnik_3[j]))
                    perf_vectorsparse_3[j].append(float(csvfile16.perf[NO_file]/mean_sputnik_3[j]))
                    perf_vectorsparse8_3[j].append(float(csvfile17.perf[NO_file]/mean_sputnik_3[j]))
                    perf_vectorsparse16_3[j].append(float(csvfile18.perf[NO_file]/mean_sputnik_3[j]))
                    perf_rospmm8_3[j].append(float(csvfile19.perf[NO_file]/mean_sputnik_3[j]))
                    perf_rospmm16_3[j].append(float(csvfile20.perf[NO_file]/mean_sputnik_3[j]))
                    # perf_cusparse_4[j].append(float(csvfile21.perf[NO_file]/mean_sputnik_4[j]))
                    perf_vectorsparse_4[j].append(float(csvfile23.perf[NO_file]/mean_sputnik_4[j]))
                    perf_vectorsparse8_4[j].append(float(csvfile24.perf[NO_file]/mean_sputnik_4[j]))
                    perf_vectorsparse16_4[j].append(float(csvfile25.perf[NO_file]/mean_sputnik_4[j]))
                    perf_rospmm8_4[j].append(float(csvfile26.perf[NO_file]/mean_sputnik_4[j]))
                    perf_rospmm16_4[j].append(float(csvfile27.perf[NO_file]/mean_sputnik_4[j]))
                    test_num += 1
for i in range(4):
    for j in range(7):
        # xlabel = sp[j]
        for k in range(96):
            NO_file = i*7*96+j*96+k
            if(int(csvfile1.nnz[NO_file])!=0):
                perf_cusparse_3[j].append(float(csvfile14.perf[NO_file]/mean_sputnik_3[j]))
                perf_cusparse_4[j].append(float(csvfile21.perf[NO_file]/mean_sputnik_4[j]))

magicube_num = 0
for i in range(4):
    for j in range(7):
        # xlabel = sp[j]
        for k in range(96):
            NO_file = i*7*96+j*96+k
            if(int(csvfile28.nnz[NO_file])!=0):
                perf_magicube_rm_88[j].append(float(csvfile28.perf[NO_file]/mean_sputnik_4[j]))
                perf_magicube_rm_168[j].append(float(csvfile29.perf[NO_file]/mean_sputnik_4[j]))
                perf_magicube_rm_1616[j].append(float(csvfile30.perf[NO_file]/mean_sputnik_4[j]))
                magicube_num += 1

print("total number of csv-file: ", filenum)
print("number of valid csv-file: ", valid_file)
print("number of test csv-file: ", test_num)
print("number of magicube csv-file: ", magicube_num)
# labels = ["sputnik" for i in range(7)]
# labels = labels * 7
# print(labels)
def AXboxplot(plt, data, bias, width, color, label):
    return plt.boxplot(data, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            showmeans=True,
            widths=width)
    
def plot(ax, data, bias, width, color):
        # geo_mean = geometric_mean(data)
        # geo_rows.append([label] + data)
        # ax.plot([1, 2, 3, 4, 5, 6, 7], data, color=color)
        return ax.boxplot(data, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], 
            notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            widths=width)

# axs[1,1].set_xticklabels(['A', 'B', 'C', 'D'])
sparsity = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]

fig, axs = plt.subplots(1, 4, figsize=(24, 6))

plt.subplot(1,4,1)
sputnik1 =  plot(axs[0], perf_sputnik_11, 0, 0.2, "steelblue")
cusparse1 = plot(axs[0], perf_cusparse_1, 0.25, 0.2, "forestgreen")
rospmm81 =  plot(axs[0], perf_vectorsparse8_1, 0.5, 0.2, "lightcoral")
rospmm161 = plot(axs[0], perf_vectorsparse16_1, 0.75, 0.2, "purple")
axs[0].set_title('V100 (Volta Architecture)',fontsize=24)
axs[0].set_ylabel("",fontsize=0)
axs[0].set(ylim=(0, 3))
axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[0].set_xticklabels(sparsity)

plt.yticks(np.arange(0, 3.1, 0.5))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel('Sparsity',fontsize=24)
plt.tick_params(labelsize=24)

plt.subplot(1,4,2)
AXboxplot(plt, perf_sputnik_22, 0, 0.2, "steelblue", "sputnik")
AXboxplot(plt, perf_cusparse_2, 0.25, 0.2, "forestgreen", "cusparse")
AXboxplot(plt, perf_rospmm8_2, 0.5, 0.2, "lightcoral", "rospmm8")
AXboxplot(plt, perf_rospmm16_2, 0.75, 0.2, "purple", "rospmm16")
axs[1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
axs[1].set_ylabel("",fontsize=0)
axs[1].set(ylim=(0, 3))
axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[1].set_xticklabels(sparsity)
plt.yticks(np.arange(0, 3.1, 0.5))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel('Sparsity',fontsize=24)
plt.tick_params(labelsize=24)

plt.subplot(1,4,3)
AXboxplot(plt, perf_sputnik_33, 0, 0.2, "steelblue", "sputnik")
AXboxplot(plt, perf_cusparse_3, 0.25, 0.2, "forestgreen", "cusparse")
AXboxplot(plt, perf_rospmm8_3, 0.5, 0.2, "lightcoral", "rospmm8")
AXboxplot(plt, perf_rospmm16_3, 0.75, 0.2, "purple", "rospmm16")
axs[2].set_title('A100 (Ampere Architecture)',fontsize=24)
axs[2].set_ylabel(" ",fontsize=10)
axs[2].set(ylim=(0, 2.5))
axs[2].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[2].set_xticklabels(sparsity)
plt.yticks(np.arange(0, 2.6, 0.5))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel(' ',fontsize=40)
plt.tick_params(labelsize=24)


plt.subplot(1,4,4)
AXboxplot(plt, perf_sputnik_44, 0, 0.2, "steelblue", "sputnik")
AXboxplot(plt, perf_cusparse_4, 0.25, 0.2, "forestgreen", "cusparse")
AXboxplot(plt, perf_rospmm8_4, 0.5, 0.2, "lightcoral", "rospmm8")
AXboxplot(plt, perf_rospmm16_4, 0.75, 0.2, "purple", "rospmm16")
axs[3].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
axs[3].set_ylabel(" ",fontsize=10)
axs[3].set(ylim=(0, 2))
axs[3].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[3].set_xticklabels(sparsity)
plt.yticks(np.arange(0, 2.6, 0.5))
plt.grid(c='grey',alpha=0.5,linestyle='--')
plt.legend([], [], frameon=False)
plt.xlabel(' ',fontsize=40)
plt.tick_params(labelsize=24)


# handles, labels = g.get_legend_handles_labels()
# print(handles, labels)
# labels = ["sputnik", "cusparse", "rospmm8", "rospmm16"]
# plt.legend(handles, labels, loc='lower center', ncol=4, fontsize=30)


fig.text(0.001, 0.5, 'Performance (GFlops)', va='center', rotation='vertical', fontsize = 28)
plt.tight_layout()
plt.savefig("boxplot.pdf", format='pdf')

