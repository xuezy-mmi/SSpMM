import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

sp = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
sparsity0 = []
sparsity1 = []
sparsity2 = []
sparsity3 = []
sparsity4 = []
sparsity5 = []
sparsity6 = []
sparsity7 = []
sparsity8 = []
sparsity9 = []
sparsity10 = []
sparsity11 = []
sparsity12 = []
sparsity13 = []
sparsity14 = []
sparsity15 = []
sparsity16 = []
sparsity17 = []
sparsity18 = []
sparsity19 = []
sparsity20 = []
sparsity21 = []
sparsity22 = []
sparsity23 = []
sparsity24 = []
sparsity25 = []
sparsity26 = []
sparsity27 = []
perf0 = []
perf1 = []
perf2 = []
perf3 = []
perf4 = []
perf5 = []
perf6 = []
perf7 = []
perf8 = []
perf9 = []
perf10 = []
perf11 = []
perf12 = []
perf13 = []
perf14 = []
perf15 = []
perf16 = []
perf17 = []
perf18 = []
perf19 = []
perf20 = []
perf21 = []
perf22 = []
perf23 = []
perf24 = []
perf25 = []
perf26 = []
perf27 = []
type0 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
type6 = []
type7 = []
type8 = []
type9 = []
type10 = []
type11 = []
type12 = []
type13 = []
type14 = []
type15 = []
type16 = []
type17 = []
type18 = []
type19 = []
type20 = []
type21 = []
type22 = []
type23 = []
type24 = []
type25 = []
type26 = []
type27 = []

filenum = len(csvfile1.nnz)
print(filenum)
for i in range (filenum):
    if(int(csvfile1.nnz[i])!=0):
        perf0.append(float(csvfile0.perf[i]))
        type0.append('cusparse')
        perf1.append(float(csvfile1.perf[i]))
        type1.append('sputnik')
        perf2.append(float(csvfile2.perf[i]))
        type2.append('vectorSparse')
        perf3.append(float(csvfile3.perf[i]))
        type3.append('vectorSparse+RM8')
        perf4.append(float(csvfile4.perf[i]))
        type4.append('vectorSparse+RM16')
        # perf5.append(float(csvfile5.perf[i]))
        # type5.append('rospmm-V8')
        # perf6.append(float(csvfile6.perf[i]))
        # type6.append('rospmm-V16')
        perf7.append(float(csvfile7.perf[i]))
        type7.append('cusparse')
        perf8.append(float(csvfile8.perf[i]))
        type8.append('sputnik')
        perf9.append(float(csvfile9.perf[i]))
        type9.append('vectorSparse')
        perf10.append(float(csvfile10.perf[i]))
        type10.append('vectorSparse+RM8')
        perf11.append(float(csvfile11.perf[i]))
        type11.append('vectorSparse+RM16')
        perf12.append(float(csvfile12.perf[i]))
        type12.append('RoSpMM-V8')
        perf13.append(float(csvfile13.perf[i]))
        type13.append('RoSpMM-V16')
        perf14.append(float(csvfile14.perf[i]))
        type14.append('cusparse')
        perf15.append(float(csvfile15.perf[i]))
        type15.append('sputnik')
        perf16.append(float(csvfile16.perf[i]))
        type16.append('vectorSparse')
        perf17.append(float(csvfile17.perf[i]))
        type17.append('vectorSparse+RM8')
        perf18.append(float(csvfile18.perf[i]))
        type18.append('vectorSparse+RM16')
        perf19.append(float(csvfile19.perf[i]))
        type19.append('RoSpMM-V8')
        perf20.append(float(csvfile20.perf[i]))
        type20.append('RoSpMM-V16')
        perf21.append(float(csvfile21.perf[i]))
        type21.append('cusparse')
        perf22.append(float(csvfile22.perf[i]))
        type22.append('sputnik')
        perf23.append(float(csvfile23.perf[i]))
        type23.append('vectorSparse')
        perf24.append(float(csvfile24.perf[i]))
        type24.append('vectorSparse+RM8')
        perf25.append(float(csvfile25.perf[i]))
        type25.append('vectorSparse+RM16')
        perf26.append(float(csvfile26.perf[i]))
        type26.append('RoSpMM-V8')
        perf27.append(float(csvfile27.perf[i]))
        type27.append('RoSpMM-V16')


for i in range(4):
    for j in range(7):
        xlabel = sp[j]
        for k in range(97):
            NO_file = i*7*97+j*97+k
            if(int(csvfile1.nnz[NO_file])!=0):
                sparsity0.append(xlabel)
                sparsity1.append(xlabel)
                sparsity2.append(xlabel)
                sparsity3.append(xlabel)
                sparsity4.append(xlabel)
                # sparsity5.append(xlabel)
                # sparsity6.append(xlabel)
                sparsity7.append(xlabel)
                sparsity8.append(xlabel)
                sparsity9.append(xlabel)
                sparsity10.append(xlabel)
                sparsity11.append(xlabel)
                sparsity12.append(xlabel)
                sparsity13.append(xlabel)
                sparsity14.append(xlabel)
                sparsity15.append(xlabel)
                sparsity16.append(xlabel)
                sparsity17.append(xlabel)
                sparsity18.append(xlabel)
                sparsity19.append(xlabel)
                sparsity20.append(xlabel)
                sparsity21.append(xlabel)
                sparsity22.append(xlabel)
                sparsity23.append(xlabel)
                sparsity24.append(xlabel)
                sparsity25.append(xlabel)
                sparsity26.append(xlabel)
                sparsity27.append(xlabel)

volta_sputnik_mean = np.mean(perf1)
volta_vectorsparse_mean = np.mean(perf2)
volta_RoSpMM8_mean = np.mean(perf3)
volta_RoSpMM16_mean = np.mean(perf4)
turing_sputnik_mean = np.mean(perf8)
turing_RoSpMM8_mean = np.mean(perf12)
turing_RoSpMM16_mean = np.mean(perf13)
ampere_sputnik_mean = np.mean(perf15)
ampere_RoSpMM8_mean = np.mean(perf19)
ampere_RoSpMM16_mean = np.mean(perf20)
ada_sputnik_mean = np.mean(perf22)
ada_RoSpMM8_mean = np.mean(perf26)
ada_RoSpMM16_mean = np.mean(perf27)

speedup8_volta = volta_RoSpMM8_mean/volta_sputnik_mean
speedup16_volta = volta_RoSpMM16_mean/volta_sputnik_mean
speedup8_turing = turing_RoSpMM8_mean/turing_sputnik_mean
speedup16_turing = turing_RoSpMM16_mean/turing_sputnik_mean
speedup8_ampere = ampere_RoSpMM8_mean/ampere_sputnik_mean
speedup16_ampere = ampere_RoSpMM16_mean/ampere_sputnik_mean
speedup8_ada = ada_RoSpMM8_mean/ada_sputnik_mean
speedup16_ada = ada_RoSpMM16_mean/ada_sputnik_mean

temp_speedup1 = volta_RoSpMM8_mean/volta_vectorsparse_mean
temp_speedup2 = volta_RoSpMM16_mean/volta_vectorsparse_mean
print("||Volta || V=8 speedup: ", speedup8_volta, "V=16 speedup: ", speedup16_volta)
print("||Volta || V=8 speedup: ", temp_speedup1, "V=16 speedup: ", temp_speedup2)
print("||Turing|| V=8 speedup: ", speedup8_turing, "V=16 speedup: ", speedup16_turing)
print("||Ampere|| V=8 speedup: ", speedup8_ampere, "V=16 speedup: ", speedup16_ampere)
print("||Ada   || V=8 speedup: ", speedup8_ada, "V=16 speedup: ", speedup16_ada)



Sparsity_Volta = sparsity0 + sparsity1 + sparsity2 + sparsity3 + sparsity4# + sparsity5 + sparsity6
Perf_Volta = perf0 + perf1 + perf2 + perf3 + perf4# + perf5 + perf6
Type_Volta = type0 + type1 + type2 + type3 + type4# + type5 + type6
Sparsity_Turing = sparsity7 + sparsity8 + sparsity9 + sparsity10 + sparsity11 + sparsity12 + sparsity13
Perf_Turing = perf7 + perf8 + perf9 + perf10 + perf11 + perf12 + perf13
Type_Turing = type7 + type8 + type9 + type10 + type11 + type12 + type13
Sparsity_Ampere = sparsity14 + sparsity15 + sparsity16 + sparsity17 + sparsity18 + sparsity19 + sparsity20
Perf_Ampere = perf14 + perf15 + perf16 + perf17 + perf18 + perf19 + perf20
Type_Ampere = type14 + type15 + type16 + type17 + type18 + type19 + type20
Sparsity_Ada = sparsity21 + sparsity22 + sparsity23 + sparsity24 + sparsity25 + sparsity26 + sparsity27
Perf_Ada = perf21 + perf22 + perf23 + perf24 + perf25 + perf26 + perf27
Type_Ada = type21 + type22 + type23 + type24 + type25 + type26 + type27

# print(len(Sparsity_Ada))
# print(len(Perf_Ada))
# print(len(Type_Ada))
df_volta = pd.DataFrame({
    'sparsity': Sparsity_Volta,
    'perf': Perf_Volta,
    'lib': Type_Volta
})
df_turing = pd.DataFrame({
    'sparsity': Sparsity_Turing,
    'perf': Perf_Turing,
    'lib': Type_Turing
})
df_ampere = pd.DataFrame({
    'sparsity': Sparsity_Ampere,
    'perf': Perf_Ampere,
    'lib': Type_Ampere
})
df_ada = pd.DataFrame({
    'sparsity': Sparsity_Ada,
    'perf': Perf_Ada,
    'lib': Type_Ada
})

df1_sputnik = pd.DataFrame({
    'sparsity': sparsity8,
    'perf': perf8,
    'lib': type8
})
df1_RoSpMM8 = pd.DataFrame({
    'sparsity': sparsity12,
    'perf': perf12,
    'lib': type12
})
df1_RoSpMM16 = pd.DataFrame({
    'sparsity': sparsity13,
    'perf': perf13,
    'lib': type13
})
df2_sputnik = pd.DataFrame({
    'sparsity': sparsity15,
    'perf': perf15,
    'lib': type15
})
df2_RoSpMM8 = pd.DataFrame({
    'sparsity': sparsity19,
    'perf': perf19,
    'lib': type19
})
df2_RoSpMM16 = pd.DataFrame({
    'sparsity': sparsity20,
    'perf': perf20,
    'lib': type20
})

df3_sputnik = pd.DataFrame({
    'sparsity': sparsity22,
    'perf': perf22,
    'lib': type22
})
df3_RoSpMM8 = pd.DataFrame({
    'sparsity': sparsity26,
    'perf': perf26,
    'lib': type26
})
df3_RoSpMM16 = pd.DataFrame({
    'sparsity': sparsity27,
    'perf': perf27,
    'lib': type27
})

sputnik_mean_values1 = df1_sputnik.groupby('sparsity')['perf'].mean()
RoSpMM8_mean_values1 = df1_RoSpMM8.groupby('sparsity')['perf'].mean()
RoSpMM16_mean_values1 = df1_RoSpMM16.groupby('sparsity')['perf'].mean()
sputnik_mean_values2 = df2_sputnik.groupby('sparsity')['perf'].mean()
RoSpMM8_mean_values2 = df2_RoSpMM8.groupby('sparsity')['perf'].mean()
RoSpMM16_mean_values2 = df2_RoSpMM16.groupby('sparsity')['perf'].mean()
sputnik_mean_values3 = df3_sputnik.groupby('sparsity')['perf'].mean()
RoSpMM8_mean_values3 = df3_RoSpMM8.groupby('sparsity')['perf'].mean()
RoSpMM16_mean_values3 = df3_RoSpMM16.groupby('sparsity')['perf'].mean()
spd81 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
spd161 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
spd82 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
spd162 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
spd83 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
spd163 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(7):
    print(sp[i])
    spd81[i] = RoSpMM8_mean_values1[i]/sputnik_mean_values1[i]
    spd161[i] = RoSpMM16_mean_values1[i]/sputnik_mean_values1[i]
    spd82[i] = RoSpMM8_mean_values2[i]/sputnik_mean_values2[i]
    spd162[i] = RoSpMM16_mean_values2[i]/sputnik_mean_values2[i]
    spd83[i] = RoSpMM8_mean_values3[i]/sputnik_mean_values3[i]
    spd163[i] = RoSpMM16_mean_values3[i]/sputnik_mean_values3[i]
    print(spd81[i])
    print(spd161[i])
    print(spd82[i])
    print(spd162[i])
    print(spd83[i])
    print(spd163[i])

fig, axs = plt.subplots(2, 2, figsize=(24, 10))
# Draw a nested boxplot to show bills by day and time
# color = {'cublas': '#025159', 'cusparse': '#26c4a5', 'vectorSparse': '#54d99f', 'vectorSparse+rowmerge': '#ace08c', 'sputnik': '#e8e284', 'rospmm-V8': '#f2b652', 'rospmm-V16': '#f26101'}
color = {'cusparse': '#4ea59f', 'sputnik': '#54d99f', 'vectorSparse': '#81d8cf', 'vectorSparse+RM8': '#dbf2c4', 'vectorSparse+RM16': '#ffb4c8', 'RoSpMM-V8': '#f2b652', 'RoSpMM-V16': '#ee6a5b'}
sns.set_theme(style="ticks")
color = dict(color)
# sns.set_style("whitegrid")
# sns.set(style="ticks")
sns.set_style("whitegrid")


plt.subplot(2,2,1)
box = sns.boxplot(x='sparsity', y='perf', hue = 'lib', data=df_volta, palette = color, ax=axs[0, 0], showfliers=True,flierprops={"marker":"o","markerfacecolor":"grey","markeredgecolor":"grey","markersize":"1"},
                showmeans=True,meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":"6"})
# sns.pointplot(x='sparsity', y='perf', hue = 'lib', data=df, estimator=np.mean, color='red', ax=box0)
axs[0, 0].set_title('V100 (Volta Architecture)',fontsize=24)
axs[0, 0].set_ylabel("",fontsize=0)
axs[0, 0].set(ylim=(0, 8000))
plt.yticks(range(0, 8001, 1000))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel('Sparsity',fontsize=24)
plt.tick_params(labelsize=24)
# for _, spine in box.spines.items():
#     spine.set_color('blue')  # Set the color of the border
#     spine.set_linewidth(2)  # Set the width of the border


plt.subplot(2,2,2)
sns.boxplot(x='sparsity', y='perf', hue = 'lib', data=df_turing, palette = color, ax=axs[0, 1], showfliers=True,flierprops={"marker":"o","markerfacecolor":"grey","markeredgecolor":"grey","markersize":"1"},
                showmeans=True,meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":"6"})
axs[0, 1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
axs[0, 1].set_ylabel("",fontsize=0)
axs[0, 1].set(ylim=(0, 8000))
plt.yticks(range(0, 8001, 1000))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel('Sparsity',fontsize=24)
plt.tick_params(labelsize=24)

# plt.grid()
plt.subplot(2,2,3)
sns.boxplot(x='sparsity', y='perf', hue = 'lib', data=df_ampere, palette = color, ax=axs[1, 0], showfliers=True,flierprops={"marker":"o","markerfacecolor":"grey","markeredgecolor":"grey","markersize":"1"},
                showmeans=True,meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":"6"})
axs[1, 0].set_title('A100 (Ampere Architecture)',fontsize=24)
axs[1, 0].set_ylabel(" ",fontsize=10)
axs[1, 0].set(ylim=(0, 10000))
plt.yticks(range(0, 10001, 2000))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel(' ',fontsize=40)
plt.tick_params(labelsize=24)

# plt.grid()
plt.subplot(2,2,4)
g = sns.boxplot(x='sparsity', y='perf', hue = 'lib', data=df_ada, palette = color, ax=axs[1, 1], showfliers=True,flierprops={"marker":"o","markerfacecolor":"grey","markeredgecolor":"grey","markersize":"1"},
                showmeans=True,meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":"6"})
axs[1, 1].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
axs[1, 1].set_ylabel(" ",fontsize=10)
axs[1, 1].set(ylim=(0, 16000))
plt.yticks(range(0, 16001, 2000))
plt.grid(c='grey',alpha=0.5,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
plt.xlabel(' ',fontsize=40)
plt.tick_params(labelsize=24)
handles, labels = g.get_legend_handles_labels()

# plt.grid()
# sns.despine(offset=0, trim=True)
sns.despine(offset=0, trim=False, top=False, right=False, left=False, bottom=False)

fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=23)

fig.text(0.001, 0.5, 'Performance (GFlops)', va='center', rotation='vertical', fontsize = 28)
plt.tight_layout()

plt.savefig("exp4.pdf", format='pdf')

# sns.set_theme(style="darkgrid")
# df = sns.load_dataset("penguins")
# sns.displot(
#     df, x="flipper_length_mm", col="species", row="sex",
#     binwidth=3, height=3, facet_kws=dict(margin_titles=True),
# )
# plt.savefig("temp.pdf", format='pdf')
