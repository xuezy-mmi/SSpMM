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
    
Pcusparse1 = "../../result-data/volta_cusparse.csv"###########
Pcublas1 = "../../result-data/volta_cublas.csv"###########
Pvectorsparse1 = "../../result-data/volta_vectorsparse.csv"
Psputnik1 = "../../result-data/volta_sputnik.csv"
Prospmm81 = "../../result-data/volta_vectorsparse_8_v0.csv"
Prospmm161 = "../../data/volta_vectorsparse_16_v0.csv"
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
Pcusparse4 = "../../data/ada_cusparse.csv"###########
Pcublas4 = "../../data/ada_cublas.csv"###########
Pvectorsparse4 = "../../data/ada_vectorsparse.csv"
Psputnik4 = "../../data/ada_sputnik.csv"
Prospmm84 = "../../data/ada_rospmm_8_v0.csv"
Prospmm164 = "../../data/ada_rospmm_16_v0.csv"

Pmagicube88 = "../../data/magicube_rowmerge_L8R8.csv"
Pmagicube168 = "../../data/magicube_rowmerge_L16R8.csv"
Pmagicube1616 = "../../data/magicube_rowmerge_L16R16.csv"

total_cusparse1, mean_cusparse1, perf_cusparse1 = readfile.read_perf96(Pcusparse1)
total_cusparse2, mean_cusparse2, perf_cusparse2 = readfile.read_perf97(Pcusparse2)
total_cusparse3, mean_cusparse3, perf_cusparse3 = readfile.read_perf96(Pcusparse3)
total_cusparse4, mean_cusparse4, perf_cusparse4 = readfile.read_perf96(Pcusparse4)
total_cublas1, mean_cublas1, perf_cublas1 = readfile.read_perf96(Pcublas1)
total_cublas2, mean_cublas2, perf_cublas2 = readfile.read_perf97(Pcublas2)
total_cublas3, mean_cublas3, perf_cublas3 = readfile.read_perf96(Pcublas3)
total_cublas4, mean_cublas4, perf_cublas4 = readfile.read_perf96(Pcublas4)
total_sputnik1, mean_sputnik1, perf_sputnik1 = readfile.read_perf97(Psputnik1)
total_sputnik2, mean_sputnik2, perf_sputnik2 = readfile.read_perf97(Psputnik2)
total_sputnik3, mean_sputnik3, perf_sputnik3 = readfile.read_perf97(Psputnik3)
total_sputnik4, mean_sputnik4, perf_sputnik4 = readfile.read_perf97(Psputnik4)
total_vs1, mean_vs1, perf_vs1 = readfile.read_perf97(Pvectorsparse1)
total_vs2, mean_vs2, perf_vs2 = readfile.read_perf97(Pvectorsparse2)
total_vs3, mean_vs3, perf_vs3 = readfile.read_perf97(Pvectorsparse3)
total_vs4, mean_vs4, perf_vs4 = readfile.read_perf97(Pvectorsparse4)
total_8ro1, mean_8ro1, perf_8ro1 = readfile.read_perf97(Prospmm81)
total_8ro2, mean_8ro2, perf_8ro2 = readfile.read_perf97(Prospmm82)
total_8ro3, mean_8ro3, perf_8ro3 = readfile.read_perf97(Prospmm83)
total_8ro4, mean_8ro4, perf_8ro4 = readfile.read_perf97(Prospmm84)
total_16ro1, mean_16ro1, perf_16ro1 = readfile.read_perf97(Prospmm161)
total_16ro2, mean_16ro2, perf_16ro2 = readfile.read_perf97(Prospmm162)
total_16ro3, mean_16ro3, perf_16ro3 = readfile.read_perf97(Prospmm163)
total_16ro4, mean_16ro4, perf_16ro4 = readfile.read_perf97(Prospmm164)
total_magicube88, mean_magicube88, perf_magicube88 = readfile.read_perf96(Pmagicube88)
total_magicube168, mean_magicube168, perf_magicube168 = readfile.read_perf96(Pmagicube168)
total_magicube1616, mean_magicube1616, perf_magicube1616 = readfile.read_perf96(Pmagicube1616)

norm_perf1 = mean_cublas1
norm_perf2 = mean_cublas2
norm_perf3 = mean_cublas3
norm_perf4 = mean_cublas4
norm_cublas1 = readfile.norm_perf(perf_cublas1, norm_perf1)
norm_cusparse1 = readfile.norm_perf(perf_cusparse1, norm_perf1)
norm_sputnik1 = readfile.norm_perf(perf_sputnik1, norm_perf1)
norm_vs1 = readfile.norm_perf(perf_vs1, norm_perf1)
norm_8ro1 = readfile.norm_perf(perf_8ro1, norm_perf1)
norm_16ro1 = readfile.norm_perf(perf_16ro1, norm_perf1)
norm_cublas2 = readfile.norm_perf(perf_cublas2, norm_perf2)
norm_cusparse2 = readfile.norm_perf(perf_cusparse2, norm_perf2)
norm_sputnik2 = readfile.norm_perf(perf_sputnik2, norm_perf2)
norm_vs2 = readfile.norm_perf(perf_vs2, norm_perf2)
norm_8ro2 = readfile.norm_perf(perf_8ro2, norm_perf2)
norm_16ro2 = readfile.norm_perf(perf_16ro2, norm_perf2)
norm_cublas3 = readfile.norm_perf(perf_cublas3, norm_perf3)
norm_cusparse3 = readfile.norm_perf(perf_cusparse3, norm_perf3)
norm_sputnik3 = readfile.norm_perf(perf_sputnik3, norm_perf3)
norm_vs3 = readfile.norm_perf(perf_vs3, norm_perf3)
norm_8ro3 = readfile.norm_perf(perf_8ro3, norm_perf3)
norm_16ro3 = readfile.norm_perf(perf_16ro3, norm_perf3)
norm_cublas4 = readfile.norm_perf(perf_cublas4, norm_perf4)
norm_cusparse4 = readfile.norm_perf(perf_cusparse4, norm_perf4)
norm_sputnik4 = readfile.norm_perf(perf_sputnik4, norm_perf4)
norm_vs4 = readfile.norm_perf(perf_vs4, norm_perf4)
norm_8ro4 = readfile.norm_perf(perf_8ro4, norm_perf4)
norm_16ro4 = readfile.norm_perf(perf_16ro4, norm_perf4)

norm_magicube88 = readfile.norm_perf(perf_magicube88, norm_perf3)
norm_magicube168 = readfile.norm_perf(perf_magicube168, norm_perf3)
norm_magicube1616 = readfile.norm_perf(perf_magicube1616, norm_perf3)

sparsity = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
# y_norm = np.ones_like(sparsity)

fig, axs = plt.subplots(2, 2, figsize=(30, 10))

norm = 1
if (norm == 0):
    plt.subplot(2,2,1)
    cublas1 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    cusparse1 = AXboxplot(plt, perf_cusparse1, 0.15, 0.1, "#54d99f")
    sputnik1 =  AXboxplot(plt, perf_sputnik1, 0.30, 0.1, "#81d8cf")
    vectorsparse1 = AXboxplot(plt, perf_vs1, 0.45, 0.1, "#dbf2c4")
    rospmm81 =  AXboxplot(plt, perf_8ro1, 0.60, 0.1, "#ffb4c8")
    rospmm161 = AXboxplot(plt, perf_16ro1, 0.75, 0.1, "#ee6a5b")
    # rospmm161 = plot(axs[0], perf_vectorsparse16_1, 0.90, 0.1, "#ffb4c8")
    axs[0,0].set_title('V100 (Volta Architecture)',fontsize=24)
    axs[0,0].set_ylabel("",fontsize=0)
    # axs[0].set(ylim=(0, 3))
    axs[0,0].set(ylim=(0, 8000))
    axs[0,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,0].set_xticklabels(sparsity)
    # plt.yticks(np.arange(0, 3.1, 0.5))
    plt.yticks(np.arange(0, 8001, 1000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,2)
    cublas2 =  AXboxplot(plt, perf_cublas2, 0, 0.1, "#4ea59f")
    cusparse2 = AXboxplot(plt, perf_cusparse2, 0.15, 0.1, "#54d99f")
    sputnik2 =  AXboxplot(plt, perf_sputnik2, 0.30, 0.1, "#81d8cf")
    vectorsparse2 = AXboxplot(plt, perf_vs2, 0.45, 0.1, "#dbf2c4")
    rospmm82 =  AXboxplot(plt, perf_8ro2, 0.60, 0.1, "#ffb4c8")
    rospmm162 = AXboxplot(plt, perf_16ro2, 0.75, 0.1, "#ee6a5b")
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
    plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,3)
    cublas3 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    cusparse3 = AXboxplot(plt, perf_cusparse3, 0.15, 0.1, "#54d99f")
    sputnik3 =  AXboxplot(plt, perf_sputnik3, 0.30, 0.1, "#81d8cf")
    vectorsparse3 = AXboxplot(plt, perf_vs3, 0.45, 0.1, "#dbf2c4")
    rospmm83 =  AXboxplot(plt, perf_8ro3, 0.60, 0.1, "#ffb4c8")
    rospmm163 = AXboxplot(plt, perf_16ro3, 0.75, 0.1, "#ee6a5b")
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
    plt.xlabel(' ',fontsize=40)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,4)
    cublas4 =  AXboxplot(plt, perf_cublas4, 0, 0.1, "#4ea59f")
    cusparse4 = AXboxplot(plt, perf_cusparse4, 0.15, 0.1, "#54d99f")
    sputnik4 =  AXboxplot(plt, perf_sputnik4, 0.30, 0.1, "#81d8cf")
    vectorsparse4 = AXboxplot(plt, perf_vs4, 0.45, 0.1, "#dbf2c4")
    rospmm84 =  AXboxplot(plt, perf_8ro4, 0.60, 0.1, "#ffb4c8")
    rospmm164 = AXboxplot(plt, perf_16ro4, 0.75, 0.1, "#ee6a5b")
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
    plt.xlabel(' ',fontsize=40)
    plt.tick_params(labelsize=24)

    fig.text(0.001, 0.5, 'Performance (GFlops)', va='center', rotation='vertical', fontsize = 28)
    plt.tight_layout()
    plt.savefig("box.pdf", format='pdf')
else:
    plt.subplot(2,2,1)
    # cublas1 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    vectorsparse1 = AXboxplot(plt, norm_vs1, 0.15, 0.1, "#dbf2c4")
    cusparse1 = AXboxplot(plt, norm_cusparse1, 0.30, 0.1, "#54d99f")
    sputnik1 =  AXboxplot(plt, norm_sputnik1, 0.45, 0.1, "#81d8cf")
    rospmm81 =  AXboxplot(plt, norm_8ro1, 0.60, 0.1, "#ffb4c8")
    rospmm161 = AXboxplot(plt, norm_16ro1, 0.75, 0.1, "#ee6a5b")
    # rospmm161 = plot(axs[0], perf_vectorsparse16_1, 0.90, 0.1, "#ffb4c8")
    axs[0,0].set_title('V100 (Volta Architecture)',fontsize=24)
    axs[0,0].set_ylabel("",fontsize=0)
    axs[0,0].set(ylim=(0, 2))
    # axs[0,0].set(ylim=(0, 8000))
    axs[0,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,0].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.1, 0.5))
    # plt.yticks(np.arange(0, 8001, 1000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,2)
    # cublas2 =  AXboxplot(plt, perf_cublas2, 0, 0.1, "#4ea59f")
    vectorsparse2 = AXboxplot(plt, norm_vs2, 0.15, 0.1, "#dbf2c4")
    cusparse2 = AXboxplot(plt, norm_cusparse2, 0.30, 0.1, "#54d99f")
    sputnik2 =  AXboxplot(plt, norm_sputnik2, 0.45, 0.1, "#81d8cf")
    rospmm82 =  AXboxplot(plt, norm_8ro2, 0.60, 0.1, "#ffb4c8")
    rospmm162 = AXboxplot(plt, norm_16ro2, 0.75, 0.1, "#ee6a5b")
    axs[0,1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
    axs[0,1].set_ylabel("",fontsize=0)
    axs[0,1].set(ylim=(0, 2))
    # axs[0,1].set(ylim=(0, 7000))
    axs[0,1].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[0,1].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.1, 0.5))
    # plt.yticks(np.arange(0, 7001, 1000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    plt.xlabel('Sparsity',fontsize=24)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,3)
    # cublas3 =  AXboxplot(plt, perf_cublas1, 0, 0.1, "#4ea59f")
    vectorsparse3 = AXboxplot(plt, norm_vs3, 0.12, 0.1, "#dbf2c4")
    cusparse3 = AXboxplot(plt, norm_cusparse3, 0.24, 0.1, "#54d99f")
    sputnik3 =  AXboxplot(plt, norm_sputnik3, 0.36, 0.1, "#81d8cf")
    magicube88 =  AXboxplot(plt, norm_magicube88, 0.48, 0.1, "#4ea59f")
    magicube168 = AXboxplot(plt, norm_magicube168, 0.60, 0.1, "#4ea59f")
    magicube1616 =  AXboxplot(plt, norm_magicube1616, 0.72, 0.1, "#4ea59f")
    rospmm83 =  AXboxplot(plt, norm_8ro3, 0.84, 0.1, "#ffb4c8")
    rospmm163 = AXboxplot(plt, norm_16ro3, 0.96, 0.1, "#ee6a5b")
    axs[1,0].set_title('A100 (Ampere Architecture)',fontsize=24)
    axs[1,0].set_ylabel(" ",fontsize=10)
    axs[1,0].set(ylim=(0, 2))
    # axs[1,0].set(ylim=(0, 10000))
    axs[1,0].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[1,0].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.1, 0.5))
    # plt.yticks(np.arange(0, 10001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    # plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
    plt.legend([], [], frameon=False)
    plt.xlabel(' ',fontsize=40)
    plt.tick_params(labelsize=24)

    plt.subplot(2,2,4)
    # cublas4 =  AXboxplot(plt, perf_cublas4, 0, 0.1, "#4ea59f")
    vectorsparse4 = AXboxplot(plt, norm_vs4, 0.15, 0.1, "#dbf2c4")
    cusparse4 = AXboxplot(plt, norm_cusparse4, 0.30, 0.1, "#54d99f")
    sputnik4 =  AXboxplot(plt, norm_sputnik4, 0.45, 0.1, "#81d8cf")
    rospmm84 =  AXboxplot(plt, norm_8ro4, 0.60, 0.1, "#ffb4c8")
    rospmm164 = AXboxplot(plt, norm_16ro4, 0.75, 0.1, "#ee6a5b")
    axs[1,1].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
    axs[1,1].set_ylabel(" ",fontsize=10)
    axs[1,1].set(ylim=(0, 2))
    # axs[1,1].set(ylim=(0, 14000))
    axs[1,1].set_xticks([1, 2, 3, 4, 5, 6, 7])
    axs[1,1].set_xticklabels(sparsity)
    plt.yticks(np.arange(0, 2.1, 0.5))
    # plt.yticks(np.arange(0, 14001, 2000))
    plt.grid(c='grey',alpha=0.5,linestyle='--')
    plt.legend([], [], frameon=False)
    plt.xlabel(' ',fontsize=40)
    plt.tick_params(labelsize=24)

    handles = [vectorsparse4["boxes"][0], cusparse4["boxes"][0], sputnik4["boxes"][0],
               magicube88["boxes"][0], magicube168["boxes"][0], magicube1616["boxes"][0], rospmm84["boxes"][0], rospmm164["boxes"][0]]
    labels = ["vectorSparse", "cuSPARSE", "sputnik", "magicube-L8R8", "magicube-L16R8", "magicubeL16R16", "RoSpMM-V8", "RoSpMM-V16"]
    
    fig.text(0.001, 0.5, 'Performance (GFlops)', va='center', rotation='vertical', fontsize = 28)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.86, top=0.95, bottom=0.04)
    plt.legend(handles, labels, bbox_to_anchor=(1.36, 1), loc='center right', ncol=1, fontsize = 28)
    plt.savefig("box_norm.pdf", format='pdf')