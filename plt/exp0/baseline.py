import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


csvfile1 = pd.read_csv('../../data/volta_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile2 = pd.read_csv('../../data/volta_vectorsparse_structured.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile3 = pd.read_csv('../../data/turing_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile4 = pd.read_csv('../../data/turing_vectorsparse_structured.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile5 = pd.read_csv('../../data/ampere_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile6 = pd.read_csv('../../data/ampere_vectorsparse_structured.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile7 = pd.read_csv('../../data/ada_sputnik.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile8 = pd.read_csv('../../data/ada_vectorsparse_structured.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
sp = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
sparsity1 = []
sparsity2 = []
sparsity3 = []
sparsity4 = []
sparsity5 = []
sparsity6 = []
sparsity7 = []
sparsity8 = []
perf1 = []
perf2 = []
perf3 = []
perf4 = []
perf5 = []
perf6 = []
perf7 = []
perf8 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
type6 = []
type7 = []
type8 = []
filenum = len(csvfile1.nnz)
print(filenum)
for i in range (filenum):
    if(int(csvfile1.nnz[i])!=0):
        perf1.append(float(csvfile1.perf[i])/1000)
        type1.append('sputnik')
        perf2.append(float(csvfile2.perf[i])/1000)
        type2.append('vectorSparse')
        perf3.append(float(csvfile3.perf[i])/1000)
        type3.append('sputnik')
        perf4.append(float(csvfile4.perf[i])/1000)
        type4.append('vectorSparse')
        perf5.append(float(csvfile5.perf[i])/1000)
        type5.append('sputnik')
        perf6.append(float(csvfile6.perf[i])/1000)
        type6.append('vectorSparse')
        perf7.append(float(csvfile7.perf[i])/1000)
        type7.append('sputnik')
        perf8.append(float(csvfile8.perf[i])/1000)
        type8.append('vectorSparse')
for i in range(4):
    for j in range(7):
        xlabel = sp[j]
        for k in range(97):
            NO_file = i*7*97+j*97+k
            if(int(csvfile1.nnz[NO_file])!=0):
                sparsity1.append(xlabel)
                sparsity2.append(xlabel)
                sparsity3.append(xlabel)
                sparsity4.append(xlabel)
                sparsity5.append(xlabel)
                sparsity6.append(xlabel)
                sparsity7.append(xlabel)
                sparsity8.append(xlabel)

df_s_volta = pd.DataFrame({
    'sparsity': sparsity1,
    'perf': perf1,
    'lib': type1
})
df_v_volta = pd.DataFrame({
    'sparsity': sparsity2,
    'perf': perf2,
    'lib': type2
})
df_s_turing = pd.DataFrame({
    'sparsity': sparsity3,
    'perf': perf3,
    'lib': type3
})
df_v_turing = pd.DataFrame({
    'sparsity': sparsity4,
    'perf': perf4,
    'lib': type4
})
df_s_ampere = pd.DataFrame({
    'sparsity': sparsity5,
    'perf': perf5,
    'lib': type5
})
df_v_ampere = pd.DataFrame({
    'sparsity': sparsity6,
    'perf': perf6,
    'lib': type6
})
df_s_ada = pd.DataFrame({
    'sparsity': sparsity7,
    'perf': perf7,
    'lib': type7
})
df_v_ada = pd.DataFrame({
    'sparsity': sparsity8,
    'perf': perf8,
    'lib': type8
})
sputnik_volta_mean = df_s_volta.groupby('sparsity')['perf'].mean()
vectorsparse_volta_mean = df_v_volta.groupby('sparsity')['perf'].mean()
sputnik_turing_mean = df_s_turing.groupby('sparsity')['perf'].mean()
vectorsparse_turing_mean = df_v_turing.groupby('sparsity')['perf'].mean()
sputnik_ampere_mean = df_s_ampere.groupby('sparsity')['perf'].mean()
vectorsparse_ampere_mean = df_v_ampere.groupby('sparsity')['perf'].mean()
sputnik_ada_mean = df_s_ada.groupby('sparsity')['perf'].mean()
vectorsparse_ada_mean = df_v_ada.groupby('sparsity')['perf'].mean()
peak_volta = [125000,125000,125000,125000,125000,125000,125000]
peak_turing = [108000,108000,108000,108000,108000,108000,108000]
peak_ampere = [312000,312000,312000,312000,312000,312000,312000]
peak_ada = [330000,330000,330000,330000,330000,330000,330000]
vectorsparse_util_volta = []
vectorsparse_util_turing = []
vectorsparse_util_ampere = []
vectorsparse_util_ada = []
sputnik_util_volta = []
sputnik_util_turing = []
sputnik_util_ampere = []
sputnik_util_ada = []
for i in range(7):
    vectorsparse_util_volta.append(vectorsparse_volta_mean[i]/125000)
    vectorsparse_util_turing.append(vectorsparse_turing_mean[i]/108000)
    vectorsparse_util_ampere.append(vectorsparse_ampere_mean[i]/312000)
    vectorsparse_util_ada.append(vectorsparse_ada_mean[i]/330000)
    sputnik_util_volta.append(sputnik_volta_mean[i]/32000)
    sputnik_util_turing.append(sputnik_turing_mean[i]/27000)
    sputnik_util_ampere.append(sputnik_ampere_mean[i]/78000)
    sputnik_util_ada.append(sputnik_ada_mean[i]/82000)

# def Perf_average(data_perf):##1940
#     average_perf = [0.0, 0.0, 0.0, 0.0, 0.0]
#     sum1 = 0.0
#     for i in range(388):#0.7
#         sum1 += data_perf[i]
#     average_perf[0] += (float)(sum1 / 388)
#     sum2 = 0.0
#     for i in range(388):#0.8
#         sum2 += data_perf[388 + i]
#     average_perf[1] += (float)(sum2 / 388)
#     sum3 = 0.0
#     for i in range(388):#0.9
#         sum3 += data_perf[388*2 + i]
#     average_perf[2] += (float)(sum3 / 388)
#     sum4 = 0.0
#     for i in range(388):#0.95
#         sum4 += data_perf[388*3 + i]
#     average_perf[3] += (float)(sum4 / 388)
#     nnz_num = 0
#     sum5 = 0.0
#     for i in range(388):#0.98
#         if(data_perf[388*4 + i] != 0.0):
#             nnz_num += 1
#             sum5 += data_perf[388*4 + i]
#     average_perf[4] += (float)(sum5 / nnz_num)
#     return average_perf

# def Utilization_of_Peak(average_perf, peak):
#     utilization = [0.0 for i in range(len(average_perf))]
#     for i in range(len(average_perf)):
#         utilization[i] = (float)(average_perf[i] / peak / 10)
#     return utilization
###################X-axis label#############################
# labels = ['0.5','0.6', '0.7', '0.8', '0.9', '0.95', '0.98']

# ######csv file row num#######
# sizeofr = 2716
# ######csv file col num#######
# sizeofc = 21

# content1 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content2 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# # content3 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content4 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# # content5 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# # content6 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]


# font3 = {'family' : 'Liberation Sans',
# 'weight' : 'normal',
# 'size'   : 18,}

# with open('tspmm_8_volta.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
#     for num in range(sizeofr):
#         content1[num] = rows[num]
# with open('tspmm_8_Turing.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
#     for num in range(sizeofr):
#         content2[num] = rows[num]
# # with open('tspmm_8_ampere.csv','r') as csvfile:
#     # reader = csv.reader(csvfile)
#     # rows = [row for row in reader]
#     # for num in range(sizeofr):
#         # content3[num] = rows[num]
# with open('tspmm_8_4090.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
#     for num in range(sizeofr):
#         content4[num] = rows[num]

# data_perf_volta = [0.0 for i in range(sizeofr)]
# data_perf_turing = [0.0 for i in range(sizeofr)]
# # data_perf_ampere = [0.0 for i in range(sizeofr)]
# data_perf_ada = [0.0 for i in range(sizeofr)]


# for num in range(sizeofr):
#     data_perf_volta[num] = (float)(content1[num][4])
#     data_perf_turing[num] = (float)(content2[num][4])
#     # data_perf_ampere[num] = (float)(content3[num][4])
#     data_perf_ada[num] = (float)(content4[num][4])

# ############process the data#########################################
# average_perf_volta = Perf_average(data_perf_volta)
# average_perf_turing = Perf_average(data_perf_turing)
# # average_perf_ampere = Perf_average(data_perf_ampere)
# average_perf_ampere = [3183.86, 2884.98, 2443.60, 1845.89, 1261.83]
# average_perf_ada = Perf_average(data_perf_ada)

# util_volta  = Utilization_of_Peak(average_perf_volta,  125)
# util_turing = Utilization_of_Peak(average_perf_turing, 114)
# util_ampere = Utilization_of_Peak(average_perf_ampere, 312)
# util_ada    = Utilization_of_Peak(average_perf_ada,    330)
#####################################################################
# print(data_perf_volta[0], data_perf_turing[0], data_perf_ada[0])
# print(average_perf_volta)
# print(average_perf_turing)
# print(average_perf_ampere)
# print(average_perf_ada)
print(sputnik_volta_mean)
print(vectorsparse_volta_mean)
print(sputnik_turing_mean)
print(vectorsparse_turing_mean)
print(sputnik_ampere_mean)
print(vectorsparse_ampere_mean)
print(sputnik_ada_mean)
print(vectorsparse_ada_mean)

x = np.arange(len(sp))  # the label locations
# y = np.arange(len(men_means))

width = 0.11  # the width of the bars

fig,ax = plt.subplots(figsize=(20, 3.8))
# fig, ax = plt.subplots(2, 1, figsize=(14, 8))
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()



# fig = plt.figure()
# ax = fig.add_subplot()
# plt.bar(x - 1.25*width, peak_volta, width,color='#D3E6F6',edgecolor='black',linewidth=2,label='sputnik volta')
# plt.bar(x - 0.25*width, peak_turing, width,color='#D9EFF2',edgecolor='black', linewidth=2,label='sputnik turing')
# plt.bar(x + 0.75*width , peak_ampere, width,color='#FDF1D6',edgecolor='black', linewidth=2, label='sputnik ampere')
# plt.bar(x + 1.75*width, peak_ada, width,color='#d45e5e',edgecolor='black',linewidth=2, label='sputnik ada')
# plt.subplot(2,1,1)
plt.bar(x - 3.5*width, vectorsparse_volta_mean, 1*width,color='#4ea59f',hatch='/',edgecolor='black',linewidth=2,label='vectorsparse-Volta')
plt.bar(x - 2.5*width, vectorsparse_turing_mean, 1*width,color='#81d8cf',hatch='/',edgecolor='black',linewidth=2,label='vectorsparse-Turing')
plt.bar(x - 1.5*width, vectorsparse_ampere_mean, 1*width,color='#ffb4c8',hatch='/',edgecolor='black', linewidth=2,label='vectorsparse-Ampere')
plt.bar(x - 0.5*width, vectorsparse_ada_mean, 1*width,color='#ee6a5b',hatch='/',edgecolor='black', linewidth=2,label='vectorsparse-Ada')

plt.bar(x + 0.5*width, sputnik_volta_mean, 1*width,color='#4ea59f',hatch='\\',edgecolor='black', linewidth=2, label='sputnik-Volta')
plt.bar(x + 1.5*width, sputnik_turing_mean, 1*width,color='#81d8cf',hatch='\\',edgecolor='black', linewidth=2, label='sputnik-Turing')
plt.bar(x + 2.5*width, sputnik_ampere_mean, 1*width,color='#ffb4c8',hatch='\\',edgecolor='black',linewidth=2, label='sputnik-Ampere')
plt.bar(x + 3.5*width, sputnik_ada_mean, 1*width,color='#ee6a5b',hatch='\\',edgecolor='black',linewidth=2, label='sputnik-Ada')

##########set the position of bars#########################
for a,b in zip(x,vectorsparse_volta_mean): ##first bar
    plt.text(a-3.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,vectorsparse_turing_mean): ##second bar
    plt.text(a-2.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,vectorsparse_ampere_mean): ##third bar
    plt.text(a-1.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,vectorsparse_ada_mean): ##forth bar
    plt.text(a-0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
    
for a,b in zip(x,sputnik_volta_mean): ##first bar
    plt.text(a+0.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,sputnik_turing_mean): ##second bar
    plt.text(a+1.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,sputnik_ampere_mean): ##third bar
    plt.text(a+2.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
for a,b in zip(x,sputnik_ada_mean): ##forth bar
    plt.text(a+3.5*width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=18)
plt.ylabel('Performance (TFLOPS)',fontsize=20)
plt.xlabel('Sparsity',fontsize=18)
plt.ylim(0,15)
# plt.title('vectorSparse-sputnik',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(sp,rotation=0, fontsize = 17)

plt.tick_params(labelsize=17)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=18,ncol=2)

fig.tight_layout()
plt.savefig('exp0.pdf',dpi=300)
# plt.show()




