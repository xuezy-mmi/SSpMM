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

def AXboxplot(plt, data, bias, width, color):
    bias = bias - 0.6
    return plt.boxplot(data, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            # showmeans=True,
            widths=width)

Pcublas1 = "../../result-data/volta_cublas.csv"    
Pcusparse1 = "../../result-data/volta_cusparse.csv"
Psputnik1 = "../../result-data/volta_sputnik.csv"
Pvectorsparse1 = "../../result-data/volta_VS_v8.csv"
Psspmm81 = "../../result-data/volta_SSpMM_v8.csv"
Psspmm161 = "../../result-data/volta_SSpMM_v16.csv"

Pcublas2 = "../../result-data/turing_cublas.csv"    
Pcusparse2 = "../../result-data/turing_cusparse.csv"
Psputnik2 = "../../result-data/turing_sputnik.csv"
Pvectorsparse2 = "../../result-data/turing_VS_v8.csv"
Psspmm82 = "../../result-data/turing_SSpMM_v8.csv"
Psspmm162 = "../../result-data/turing_SSpMM_v16.csv"

Pcublas3 = "../../result-data/ampere_cublas.csv"    
Pcusparse3 = "../../result-data/ampere_cusparse.csv"
Psputnik3 = "../../result-data/ampere_sputnik.csv"
Pvectorsparse3 = "../../result-data/ampere_VS_v8.csv"
Psspmm83 = "../../result-data/ampere_SSpMM_v8.csv"
Psspmm163 = "../../result-data/ampere_SSpMM_v16.csv"

Pcublas4 = "../../result-data/ada_cublas.csv"    
Pcusparse4 = "../../result-data/ada_cusparse.csv"
Psputnik4 = "../../result-data/ada_sputnik.csv"
Pvectorsparse4 = "../../result-data/ada_VS_v8.csv"
Psspmm84 = "../../result-data/ada_SSpMM_v8.csv"
Psspmm164 = "../../result-data/ada_SSpMM_v16.csv"

Pmagicube88 = "../../data/magicube_rowmerge_L8R8.csv"
Pmagicube168 = "../../data/magicube_rowmerge_L16R8.csv"
Pmagicube1616 = "../../data/magicube_rowmerge_L16R16.csv"

suitpath_cublas1 = "../../result-data/volta_suit_cublas.csv"
suitpath_cublas2 = "../../result-data/turing_suit_cublas.csv"
suitpath_cublas3 = "../../result-data/ampere_suit_cublas.csv"
suitpath_cublas4 = "../../result-data/ada_suit_cublas.csv"
suitpath_cusparse1 = "../../result-data/volta_suit_cusparse.csv"
suitpath_cusparse2 = "../../result-data/turing_suit_cusparse.csv"
suitpath_cusparse3 = "../../result-data/ampere_suit_cusparse.csv"
suitpath_cusparse4 = "../../result-data/ada_suit_cusparse.csv"
suitpath_sputnik1 = "../../result-data/volta_suit_sputnik.csv"
suitpath_sputnik2 = "../../result-data/turing_suit_sputnik.csv"
suitpath_sputnik3 = "../../result-data/ampere_suit_sputnik.csv"
suitpath_sputnik4 = "../../result-data/ada_suit_sputnik.csv"
suitpath_SSpMM81 = "../../result-data/volta_suit_SSpMM_v8.csv"
suitpath_SSpMM82 = "../../result-data/turing_suit_SSpMM_v8.csv"
suitpath_SSpMM83 = "../../result-data/ampere_suit_SSpMM_v8.csv"
suitpath_SSpMM84 = "../../result-data/ada_suit_SSpMM_v8.csv"
suitpath_SSpMM161 = "../../result-data/volta_suit_SSpMM_v16.csv"
suitpath_SSpMM162 = "../../result-data/turing_suit_SSpMM_v16.csv"
suitpath_SSpMM163 = "../../result-data/ampere_suit_SSpMM_v16.csv"
suitpath_SSpMM164 = "../../result-data/ada_suit_SSpMM_v16.csv"

total_cusparse1, mean_cusparse1, perf_cusparse1 = readfile.read_perf(Pcusparse1)
total_cusparse2, mean_cusparse2, perf_cusparse2 = readfile.read_perf(Pcusparse2)
total_cusparse3, mean_cusparse3, perf_cusparse3 = readfile.read_perf(Pcusparse3)
total_cusparse4, mean_cusparse4, perf_cusparse4 = readfile.read_perf(Pcusparse4)
total_cublas1, mean_cublas1, perf_cublas1 = readfile.read_perf(Pcublas1)
total_cublas2, mean_cublas2, perf_cublas2 = readfile.read_perf(Pcublas2)
total_cublas3, mean_cublas3, perf_cublas3 = readfile.read_perf(Pcublas3)
total_cublas4, mean_cublas4, perf_cublas4 = readfile.read_perf(Pcublas4)
total_sputnik1, mean_sputnik1, perf_sputnik1 = readfile.read_perf(Psputnik1)
total_sputnik2, mean_sputnik2, perf_sputnik2 = readfile.read_perf(Psputnik2)
total_sputnik3, mean_sputnik3, perf_sputnik3 = readfile.read_perf(Psputnik3)
total_sputnik4, mean_sputnik4, perf_sputnik4 = readfile.read_perf(Psputnik4)
total_vs1, mean_vs1, perf_vs1 = readfile.read_perf(Pvectorsparse1)
total_vs2, mean_vs2, perf_vs2 = readfile.read_perf(Pvectorsparse2)
total_vs3, mean_vs3, perf_vs3 = readfile.read_perf(Pvectorsparse3)
total_vs4, mean_vs4, perf_vs4 = readfile.read_perf(Pvectorsparse4)

total_SSpMM1, mean_SSpMM1, perf_SSpMM1 = readfile.read_perf_SSpMM(Psspmm81, Psspmm161)
total_SSpMM2, mean_SSpMM2, perf_SSpMM2 = readfile.read_perf_SSpMM(Psspmm82, Psspmm162)
total_SSpMM3, mean_SSpMM3, perf_SSpMM3 = readfile.read_perf_SSpMM(Psspmm83, Psspmm163)
total_SSpMM4, mean_SSpMM4, perf_SSpMM4 = readfile.read_perf_SSpMM(Psspmm84, Psspmm164)

total_magicube88, mean_magicube88, perf_magicube88 = readfile.read_perf4(Pmagicube88)
total_magicube168, mean_magicube168, perf_magicube168 = readfile.read_perf4(Pmagicube168)
total_magicube1616, mean_magicube1616, perf_magicube1616 = readfile.read_perf4(Pmagicube1616)

speedup_to_cusparse1 = total_SSpMM1 / total_cusparse1
speedup_to_cusparse2 = total_SSpMM2 / total_cusparse2
speedup_to_cusparse3 = total_SSpMM3 / total_cusparse3
speedup_to_cusparse4 = total_SSpMM4 / total_cusparse4

speedup_to_vs1 = total_SSpMM1 / total_vs1
speedup_to_vs2 = total_SSpMM2 / total_vs2
speedup_to_vs3 = total_SSpMM3 / total_vs3
speedup_to_vs4 = total_SSpMM4 / total_vs4

speedup_to_sputnik1 = total_SSpMM1 / total_sputnik1
speedup_to_sputnik2 = total_SSpMM2 / total_sputnik2
speedup_to_sputnik3 = total_SSpMM3 / total_sputnik3
speedup_to_sputnik4 = total_SSpMM4 / total_sputnik4

print("Volta Speedup:")
print("to cusparse: ", speedup_to_cusparse1, "to vectorSparse: ", speedup_to_vs1, "to Sputnik: ", speedup_to_sputnik1)
print("Turing Speedup:")
print("to cusparse: ", speedup_to_cusparse2, "to vectorSparse: ", speedup_to_vs2, "to Sputnik: ", speedup_to_sputnik2)
print("Ampere Speedup:")
print("to cusparse: ", speedup_to_cusparse3, "to vectorSparse: ", speedup_to_vs3, "to Sputnik: ", speedup_to_sputnik3)
print("Ada Speedup:")
print("to cusparse: ", speedup_to_cusparse4, "to vectorSparse: ", speedup_to_vs4, "to Sputnik: ", speedup_to_sputnik4)

suitperf_cublas1 = readfile.read_suit_perf(suitpath_cublas1)
suitperf_cublas2 = readfile.read_suit_perf(suitpath_cublas2)
suitperf_cublas3 = readfile.read_suit_perf(suitpath_cublas3)
suitperf_cublas4 = readfile.read_suit_perf(suitpath_cublas4)
suitperf_cusparse1 = readfile.read_suit_perf(suitpath_cusparse1)
suitperf_cusparse2 = readfile.read_suit_perf(suitpath_cusparse2)
suitperf_cusparse3 = readfile.read_suit_perf(suitpath_cusparse3)
suitperf_cusparse4 = readfile.read_suit_perf(suitpath_cusparse4)
suitperf_sputnik1 = readfile.read_suit_perf(suitpath_sputnik1)
suitperf_sputnik2 = readfile.read_suit_perf(suitpath_sputnik2)
suitperf_sputnik3 = readfile.read_suit_perf(suitpath_sputnik3)
suitperf_sputnik4 = readfile.read_suit_perf(suitpath_sputnik4)
suitperf_SSpMM1 = readfile.read_suit_SSpMMperf(suitpath_SSpMM81, suitpath_SSpMM161)
suitperf_SSpMM2 = readfile.read_suit_SSpMMperf(suitpath_SSpMM82, suitpath_SSpMM162)
suitperf_SSpMM3 = readfile.read_suit_SSpMMperf(suitpath_SSpMM83, suitpath_SSpMM163)
suitperf_SSpMM4 = readfile.read_suit_SSpMMperf(suitpath_SSpMM84, suitpath_SSpMM164)

speedup_to_cusparse1_suit = sum(suitperf_SSpMM1) / sum(suitperf_cusparse1)
speedup_to_cusparse2_suit = sum(suitperf_SSpMM2) / sum(suitperf_cusparse2)
speedup_to_cusparse3_suit = sum(suitperf_SSpMM3) / sum(suitperf_cusparse3)
speedup_to_cusparse4_suit = sum(suitperf_SSpMM4) / sum(suitperf_cusparse4)

speedup_to_sputnik1_suit = sum(suitperf_SSpMM1) / sum(suitperf_sputnik1)
speedup_to_sputnik2_suit = sum(suitperf_SSpMM2) / sum(suitperf_sputnik2)
speedup_to_sputnik3_suit = sum(suitperf_SSpMM3) / sum(suitperf_sputnik3)
speedup_to_sputnik4_suit = sum(suitperf_SSpMM4) / sum(suitperf_sputnik4)

print("Volta Speedup SuiteSparse:")
print("to cusparse: ", speedup_to_cusparse1_suit, "to Sputnik: ", speedup_to_sputnik1_suit)
print("Turing Speedup SuiteSparse:")
print("to cusparse: ", speedup_to_cusparse2_suit, "to Sputnik: ", speedup_to_sputnik2_suit)
print("Ampere Speedup SuiteSparse:")
print("to cusparse: ", speedup_to_cusparse3_suit, "to Sputnik: ", speedup_to_sputnik3_suit)
print("Ada Speedup SuiteSparse:")
print("to cusparse: ", speedup_to_cusparse4_suit, "to Sputnik: ", speedup_to_sputnik4_suit)

norm_perf1 = mean_cublas1
norm_perf2 = mean_cublas2
norm_perf3 = mean_cublas3
norm_perf4 = mean_cublas4
norm_cublas1 = readfile.norm_perf(perf_cublas1, norm_perf1)
norm_cusparse1 = readfile.norm_perf(perf_cusparse1, norm_perf1)
norm_sputnik1 = readfile.norm_perf(perf_sputnik1, norm_perf1)
norm_vs1 = readfile.norm_perf(perf_vs1, norm_perf1)
norm_sspmm1 = readfile.norm_perf(perf_SSpMM1, norm_perf1)
norm_cublas2 = readfile.norm_perf(perf_cublas2, norm_perf2)
norm_cusparse2 = readfile.norm_perf(perf_cusparse2, norm_perf2)
norm_sputnik2 = readfile.norm_perf(perf_sputnik2, norm_perf2)
norm_vs2 = readfile.norm_perf(perf_vs2, norm_perf2)
norm_sspmm2 = readfile.norm_perf(perf_SSpMM2, norm_perf2)
norm_cublas3 = readfile.norm_perf(perf_cublas3, norm_perf3)
norm_cusparse3 = readfile.norm_perf(perf_cusparse3, norm_perf3)
norm_sputnik3 = readfile.norm_perf(perf_sputnik3, norm_perf3)
norm_vs3 = readfile.norm_perf(perf_vs3, norm_perf3)
norm_sspmm3 = readfile.norm_perf(perf_SSpMM3, norm_perf3)
norm_cublas4 = readfile.norm_perf(perf_cublas4, norm_perf4)
norm_cusparse4 = readfile.norm_perf(perf_cusparse4, norm_perf4)
norm_sputnik4 = readfile.norm_perf(perf_sputnik4, norm_perf4)
norm_vs4 = readfile.norm_perf(perf_vs4, norm_perf4)
norm_sspmm4 = readfile.norm_perf(perf_SSpMM4, norm_perf4)

norm_magicube88 = readfile.norm_perf(perf_magicube88, norm_perf3)
norm_magicube168 = readfile.norm_perf(perf_magicube168, norm_perf3)
norm_magicube1616 = readfile.norm_perf(perf_magicube1616, norm_perf3)

sparsity = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]

fig, axs = plt.subplots(4, 2, figsize=(30, 15), gridspec_kw={'width_ratios': [1,1.6]})

suit_num = 15
x = np.arange(suit_num)
x_label = [str(i+1) for i in range(suit_num)]
width = 0.21

norm = 1
if (norm == 0):
    plt.subplot(2,2,1)
    cublas1 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    cusparse1 = AXboxplot(plt, perf_cusparse1, 0.15, 0.1, "#54d99f")
    sputnik1 =  AXboxplot(plt, perf_sputnik1, 0.30, 0.1, "#81d8cf")
    vectorsparse1 = AXboxplot(plt, perf_vs1, 0.45, 0.1, "#dbf2c4")
    # rospmm81 =  AXboxplot(plt, perf_8ro1, 0.60, 0.1, "#ffb4c8")
    sspmm1 = AXboxplot(plt, perf_SSpMM1, 0.6, 0.1, "#ee6a5b")
    # rospmm161 = plot(axs[0], perf_vectorsparse16_1, 0.90, 0.1, "#ffb4c8")
    axs[0,0].set_title('V100 (Volta Architecture)',fontsize=24)
    axs[0,0].set_ylabel("",fontsize=0)
    # axs[0].set(ylim=(0, 3))
    axs[0,0].set(ylim=(0, 6000))
    axs[0,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,0].set_xticklabels(sparsity)
    # plt.yticks(np.arange(0, 3.1, 0.5))
    plt.yticks(np.arange(0, 12001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,2)
    cublas2 =  AXboxplot(plt, perf_cublas2, 0, 0.1, "#4ea59f")
    cusparse2 = AXboxplot(plt, perf_cusparse2, 0.15, 0.1, "#54d99f")
    sputnik2 =  AXboxplot(plt, perf_sputnik2, 0.30, 0.1, "#81d8cf")
    vectorsparse2 = AXboxplot(plt, perf_vs2, 0.45, 0.1, "#dbf2c4")
    # rospmm82 =  AXboxplot(plt, perf_8ro2, 0.60, 0.1, "#ffb4c8")
    sspmm2 = AXboxplot(plt, perf_SSpMM2, 0.6, 0.1, "#ee6a5b")
    axs[0,1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
    axs[0,1].set_ylabel("",fontsize=0)
    # axs[1].set(ylim=(0, 3))
    axs[0,1].set(ylim=(0, 7000))
    axs[0,1].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,1].set_xticklabels(sparsity)
    # plt.yticks(np.arange(0, 3.1, 0.5))
    plt.yticks(np.arange(0, 7001, 1000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,3)
    cublas3 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    cusparse3 = AXboxplot(plt, perf_cusparse3, 0.15, 0.1, "#54d99f")
    sputnik3 =  AXboxplot(plt, perf_sputnik3, 0.30, 0.1, "#81d8cf")
    vectorsparse3 = AXboxplot(plt, perf_vs3, 0.45, 0.1, "#dbf2c4")
    # rospmm83 =  AXboxplot(plt, perf_8ro3, 0.60, 0.1, "#ffb4c8")
    sspmm3 = AXboxplot(plt, perf_SSpMM3, 0.6, 0.1, "#ee6a5b")
    axs[1,0].set_title('A100 (Ampere Architecture)',fontsize=24)
    axs[1,0].set_ylabel(" ",fontsize=10)
    # axs[2].set(ylim=(0, 2.5))
    axs[1,0].set(ylim=(0, 10000))
    axs[1,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[1,0].set_xticklabels(sparsity)
    # plt.yticks(np.arange(0, 2.6, 0.5))
    plt.yticks(np.arange(0, 10001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,4)
    cublas4 =  AXboxplot(plt, perf_cublas4, 0, 0.1, "#4ea59f")
    cusparse4 = AXboxplot(plt, perf_cusparse4, 0.15, 0.1, "#54d99f")
    sputnik4 =  AXboxplot(plt, perf_sputnik4, 0.30, 0.1, "#81d8cf")
    vectorsparse4 = AXboxplot(plt, perf_vs4, 0.45, 0.1, "#dbf2c4")
    # rospmm84 =  AXboxplot(plt, perf_8ro4, 0.60, 0.1, "#ffb4c8")
    sspmm4 = AXboxplot(plt, perf_SSpMM4, 0.6, 0.1, "#ee6a5b")
    axs[1,1].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
    axs[1,1].set_ylabel(" ",fontsize=10)
    # axs[1,1].set(ylim=(0, 2))
    axs[1,1].set(ylim=(0, 14000))
    axs[1,1].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[1,1].set_xticklabels(sparsity)
    # plt.yticks(np.arange(0, 2.6, 0.5))
    plt.yticks(np.arange(0, 14001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    fig.text(0.001, 0.5, 'Performance (GFLOPS)', va='center', rotation='vertical', fontsize = 28)
    plt.tight_layout()
    plt.savefig("exp4.pdf", format='pdf')
else:

    plt.subplot(4,2,1)
    # cublas1 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    vectorsparse1 = AXboxplot(plt, norm_vs1, 0.15, 0.1, "#dbf2c4")
    cusparse1 = AXboxplot(plt, norm_cusparse1, 0.30, 0.1, "#54d99f")
    sputnik1 =  AXboxplot(plt, norm_sputnik1, 0.45, 0.1, "#4ea59f")
    # rospmm81 =  AXboxplot(plt, norm_8ro1, 0.60, 0.1, "#ffb4c8")
    sspmm1 = AXboxplot(plt, norm_sspmm1, 0.6, 0.1, "#ee6a5b")
    # rospmm161 = plot(axs[0], perf_vectorsparse16_1, 0.90, 0.1, "#ffb4c8")
    axs[0,0].set_ylabel(" ",fontsize=20)
    axs[0,0].set(ylim=(0, 2))
    axs[0,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,0].set_xticklabels(sparsity)
    axs[0,0].set_title('Deep Learning Matrix Collection',fontsize=24)
    plt.yticks(np.arange(0, 2.1, 0.5))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    plt.xlabel(' ',fontsize=5)
    plt.tick_params(labelsize=24)

    plt.subplot(4,2,2)
    plt.bar(x - 3*width, suitperf_cublas1, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='cuBLAS')
    plt.bar(x - 2*width, suitperf_cusparse1, width,color='#54d99f',edgecolor='black',linewidth=1.5,label='cuSPARSE')
    plt.bar(x - 1*width, suitperf_sputnik1, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='sputnik')
    plt.bar(x          , suitperf_SSpMM1, width,color='#ee6a5b',edgecolor='black', linewidth=1.5,label='SSpMM')
    for a,b in zip(x,suitperf_cublas1): ##label position
        plt.text(a - 3*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_cusparse1): ##label position
        plt.text(a - 2*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_sputnik1): ##label position
        plt.text(a - 1*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_SSpMM1): ##label position
        plt.text(a          ,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(" ",fontsize=20)
    plt.yticks(range(0, 6001, 1000))
    # plt.title('Scores by group and gender',fontsize=28)
    axs[0,1].set(ylim=(0, 6000))
    axs[0,1].set_xticks(x)
    axs[0,1].set_xticklabels(x_label,rotation=0, fontsize = 22)
    axs[0,1].set_title('SuiteSparse Matrix Collection',fontsize=24)
    plt.xlabel(' ',fontsize=5)
    plt.tick_params(labelsize=22)
    plt.grid(c='grey',alpha=0.9,linestyle='--')
    # plt.legend(loc="upper center", borderaxespad=0,fontsize=22,ncol=2)

    plt.subplot(4,2,3)
    # cublas2 =  AXboxplot(plt, perf_cublas2, 0, 0.1, "#4ea59f")
    vectorsparse2 = AXboxplot(plt, norm_vs2, 0.15, 0.1, "#dbf2c4")
    cusparse2 = AXboxplot(plt, norm_cusparse2, 0.30, 0.1, "#54d99f")
    sputnik2 =  AXboxplot(plt, norm_sputnik2, 0.45, 0.1, "#4ea59f")
    # rospmm82 =  AXboxplot(plt, norm_8ro2, 0.60, 0.1, "#ffb4c8")
    sspmm2 = AXboxplot(plt, norm_sspmm2, 0.6, 0.1, "#ee6a5b")
    # axs[0,1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
    axs[1,0].set_ylabel(" ",fontsize=20)
    axs[1,0].set(ylim=(0, 2.7))
    # axs[0,1].set(ylim=(0, 7000))
    axs[1,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[1,0].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.6, 0.5))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(4,2,4)
    plt.bar(x - 3*width, suitperf_cublas2, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='cuBLAS')
    plt.bar(x - 2*width, suitperf_cusparse2, width,color='#54d99f',edgecolor='black',linewidth=1.5,label='cuSPARSE')
    plt.bar(x - 1*width, suitperf_sputnik2, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='sputnik')
    plt.bar(x          , suitperf_SSpMM2, width,color='#ee6a5b',edgecolor='black', linewidth=1.5,label='SSpMM')
    for a,b in zip(x,suitperf_cublas2): ##label position
        plt.text(a - 3*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_cusparse2): ##label position
        plt.text(a - 2*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_sputnik2): ##label position
        plt.text(a - 1*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_SSpMM2): ##label position
        plt.text(a          ,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(" ",fontsize=20)
    plt.yticks(range(0, 6001, 1000))
    # plt.title('Scores by group and gender',fontsize=28)
    axs[1,1].set(ylim=(0, 6200))
    axs[1,1].set_xticks(x)
    axs[1,1].set_xticklabels(x_label,rotation=0, fontsize = 22)
    # plt.xlabel('Matrix ID',fontsize=25)
    plt.tick_params(labelsize=22)
    plt.grid(c='grey',alpha=0.9,linestyle='--')
    
    plt.subplot(4,2,5)
    # cublas3 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    vectorsparse3 = AXboxplot(plt, norm_vs3, 0.12, 0.1, "#dbf2c4")
    cusparse3 = AXboxplot(plt, norm_cusparse3, 0.24, 0.1, "#54d99f")
    sputnik3 =  AXboxplot(plt, norm_sputnik3, 0.36, 0.1, "#4ea59f")
    
    color = "#6c5e82"
    bias = 0.48-0.6
    magicube88 = plt.boxplot(norm_magicube88, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color, hatch='\\'),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            # showmeans=True,
            widths=0.1)
    bias = 0.6-0.6
    magicube168 = plt.boxplot(norm_magicube168, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color, hatch='/'),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            widths=0.1)
    bias = 0.72-0.6
    magicube1616 = plt.boxplot(norm_magicube1616, positions=[1 + bias, 2 + bias, 3 + bias, 4 + bias, 5 + bias, 6 + bias, 7 + bias], notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color='black'),
            # showmeans=True,
            widths=0.1)
    # magicube88 =  AXboxplot(plt, norm_magicube88, 0.48, 0.1, "#6c5e82")
    # magicube168 = AXboxplot(plt, norm_magicube168, 0.60, 0.1, "#6c5e82")
    # magicube1616 =  AXboxplot(plt, norm_magicube1616, 0.72, 0.1, "#6c5e82")
    sspmm3 = AXboxplot(plt, norm_sspmm3, 0.84, 0.1, "#ee6a5b")
    # axs[1,0].set_title('A100 (Ampere Architecture)',fontsize=24)
    axs[2,0].set_ylabel(" ",fontsize=20)
    axs[2,0].set(ylim=(0, 1.5))
    axs[2,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[2,0].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 1.6, 0.5))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    # plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(4,2,6)
    plt.bar(x - 3*width, suitperf_cublas3, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='cuBLAS')
    plt.bar(x - 2*width, suitperf_cusparse3, width,color='#54d99f',edgecolor='black',linewidth=1.5,label='cuSPARSE')
    plt.bar(x - 1*width, suitperf_sputnik3, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='sputnik')
    plt.bar(x          , suitperf_SSpMM3, width,color='#ee6a5b',edgecolor='black', linewidth=1.5,label='SSpMM')
    for a,b in zip(x,suitperf_cublas3): ##label position
        plt.text(a - 3*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_cusparse3): ##label position
        plt.text(a - 2*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_sputnik3): ##label position
        plt.text(a - 1*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_SSpMM3): ##label position
        plt.text(a          ,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(" ",fontsize=20)
    plt.yticks(range(0, 8001, 2000))
    axs[2,1].set(ylim=(0, 8000))
    axs[2,1].set_xticks(x)
    axs[2,1].set_xticklabels(x_label,rotation=0, fontsize = 22)
    # plt.xlabel('Matrix ID',fontsize=25)
    plt.tick_params(labelsize=22)
    plt.grid(c='grey',alpha=0.9,linestyle='--')
    
    plt.subplot(4,2,7)
    # cublas4 =  AXboxplot(plt, perf_cublas4, 0, 0.1, "#4ea59f")
    vectorsparse4 = AXboxplot(plt, norm_vs4, 0.15, 0.1, "#dbf2c4")
    cusparse4 = AXboxplot(plt, norm_cusparse4, 0.30, 0.1, "#54d99f")
    sputnik4 =  AXboxplot(plt, norm_sputnik4, 0.45, 0.1, "#4ea59f")
    # rospmm84 =  AXboxplot(plt, norm_8ro4, 0.60, 0.1, "#ffb4c8")
    sspmm4 = AXboxplot(plt, norm_sspmm4, 0.6, 0.1, "#ee6a5b")
    # axs[1,1].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
    axs[3,0].set_ylabel(" ",fontsize=20)
    axs[3,0].set(ylim=(0, 2))
    # axs[1,1].set(ylim=(0, 14000))
    axs[3,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[3,0].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.1, 0.5))
    # plt.yticks(np.arange(0, 14001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    plt.legend([], [], frameon=False)
    plt.xlabel(' ',fontsize=90)
    plt.tick_params(labelsize=24)

    plt.subplot(4,2,8)
    cublas4 = plt.bar(x - 3*width, suitperf_cublas4, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='cuBLAS')
    plt.bar(x - 2*width, suitperf_cusparse4, width,color='#54d99f',edgecolor='black',linewidth=1.5,label='cuSPARSE')
    plt.bar(x - 1*width, suitperf_sputnik4, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='sputnik')
    plt.bar(x          , suitperf_SSpMM4, width,color='#ee6a5b',edgecolor='black', linewidth=1.5,label='SSpMM')
    for a,b in zip(x,suitperf_cublas4): ##label position
        plt.text(a - 3*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_cusparse4): ##label position
        plt.text(a - 2*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_sputnik4): ##label position
        plt.text(a - 1*width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    for a,b in zip(x,suitperf_SSpMM4): ##label position
        plt.text(a          ,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=19)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel(" ",fontsize=20)
    plt.yticks(range(0, 14001, 2000))
    # plt.title('Scores by group and gender',fontsize=28)
    axs[3,1].set(ylim=(0, 14000))
    axs[3,1].set_xticks(x)
    axs[3,1].set_xticklabels(x_label,rotation=0, fontsize = 22)
    plt.xlabel(' ',fontsize=90)
    plt.tick_params(labelsize=22)
    plt.grid(c='grey',alpha=0.9,linestyle='--')
    
    handles = [cublas4[0]] + [vectorsparse4["boxes"][0], cusparse4["boxes"][0], sputnik4["boxes"][0],
               magicube88["boxes"][0], magicube168["boxes"][0], magicube1616["boxes"][0], sspmm4["boxes"][0]]
    labels = ["cuBLAS", "vectorSparse", "cuSPARSE", "sputnik", "magicube-L8R8", "magicube-L16R8", "magicubeL16R16", "SSpMM"]
    
    fig.text(0.001, 0.5, 'Speedup against cuBLAS', va='center', rotation='vertical', fontsize = 28)
    fig.text(0.39, 0.5, 'Performance (GFLOPS)', va='center', rotation='vertical', fontsize = 28)
    
    fig.text(0.04, 0.95, 'V100(Volta)', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.44, 0.95, 'V100', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.04, 0.72, '2080Ti(Turing)', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.44, 0.72, '2080Ti', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.04, 0.495, 'A100(Ampere)', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.44, 0.495, 'A100', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.04, 0.27, '4090(Ada)', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.44, 0.27, '4090', va='center', rotation='horizontal', fontsize = 28)
    fig.text(0.18, 0.755, 'Sparsity', va='center', rotation='horizontal', fontsize = 24)
    fig.text(0.68, 0.755, 'Matrix ID', va='center', rotation='horizontal', fontsize = 24)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.04, right=0.86, top=0.95, bottom=0.04)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize = 28)
    plt.savefig("exp4.pdf", format='pdf')