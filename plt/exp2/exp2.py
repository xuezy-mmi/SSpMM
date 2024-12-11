import matplotlib.pyplot as plt
# import numpy as np
import csv
import math
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


font3 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 18,}

# csvfile0  = pd.read_csv('../../data/volta_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile1  = pd.read_csv('../../data/volta_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile2  = pd.read_csv('../../data/volta_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile3  = pd.read_csv('../../data/volta_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

# csvfile4  = pd.read_csv('../../data/turing_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile5  = pd.read_csv('../../data/turing_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile6  = pd.read_csv('../../data/turing_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile7  = pd.read_csv('../../data/turing_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

# csvfile8  = pd.read_csv('../../data/ampere_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile9  = pd.read_csv('../../data/ampere_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile10 = pd.read_csv('../../data/ampere_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile11 = pd.read_csv('../../data/ampere_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

# csvfile12 = pd.read_csv('../../data/ada_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile13 = pd.read_csv('../../data/ada_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile14 = pd.read_csv('../../data/ada_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
# csvfile15 = pd.read_csv('../../data/ada_rospmm_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

csv1 = pd.read_csv('../../result-data/volta_VS_v1.csv',usecols=[2],names=["perf"])
csv2 = pd.read_csv('../../result-data/volta_VS_v8.csv',usecols=[2],names=["perf"])
csv3 = pd.read_csv('../../result-data/volta_WMMASpMM_v8.csv',usecols=[2],names=["perf"])
csv4 = pd.read_csv('../../result-data/volta_SSpMM_v8.csv',usecols=[0,4],names=["nnz","perf"])

csv5 = pd.read_csv('../../result-data/turing_VS_v1.csv',usecols=[2],names=["perf"])
csv6 = pd.read_csv('../../result-data/turing_VS_v8.csv',usecols=[2],names=["perf"])
csv7 = pd.read_csv('../../result-data/turing_WMMASpMM_v8.csv',usecols=[2],names=["perf"])
csv8 = pd.read_csv('../../result-data/turing_SSpMM_v8.csv',usecols=[0,4],names=["nnz","perf"])

csv9  = pd.read_csv('../../result-data/ampere_VS_v1.csv',usecols=[2],names=["perf"])
csv10 = pd.read_csv('../../result-data/ampere_VS_v8.csv',usecols=[2],names=["perf"])
csv11 = pd.read_csv('../../result-data/ampere_WMMASpMM_v8.csv',usecols=[2],names=["perf"])
csv12 = pd.read_csv('../../result-data/ampere_SSpMM_v8.csv',usecols=[0,4],names=["nnz","perf"])

csv13 = pd.read_csv('../../result-data/ada_VS_v1.csv',usecols=[2],names=["perf"])
csv14 = pd.read_csv('../../result-data/ada_VS_v8.csv',usecols=[2],names=["perf"])
csv15 = pd.read_csv('../../result-data/ada_WMMASpMM_v8.csv',usecols=[2],names=["perf"])
csv16 = pd.read_csv('../../result-data/ada_SSpMM_v8.csv',usecols=[0,4],names=["nnz","perf"])

sp = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
sparsity = []
perf_list = [[] for i in range(4*4*3)]

type1 = []
type2 = []
type3 = []
type4 = []


filenum = len(csv4.nnz)
print(filenum)
valid_num = 0
for i in range (2688):
    if(int(csv4.nnz[i])!=0):
        valid_num += 1
        perf_list[0 ].append(float(csv1.perf[i])/1000)
        perf_list[1 ].append(float(csv2.perf[i])/1000)
        perf_list[2 ].append(float(csv3.perf[i])/1000)
        perf_list[3 ].append(float(csv4.perf[i])/1000)
        perf_list[4 ].append(float(csv5.perf[i])/1000)
        perf_list[5 ].append(float(csv6.perf[i])/1000)
        perf_list[6 ].append(float(csv7.perf[i])/1000)
        perf_list[7 ].append(float(csv8.perf[i])/1000)
        perf_list[8 ].append(float(csv9.perf[i])/1000)
        perf_list[9 ].append(float(csv10.perf[i])/1000)
        perf_list[10].append(float(csv11.perf[i])/1000)
        perf_list[11].append(float(csv12.perf[i])/1000)
        perf_list[12].append(float(csv13.perf[i])/1000)
        perf_list[13].append(float(csv14.perf[i])/1000)
        perf_list[14].append(float(csv15.perf[i])/1000)
        perf_list[15].append(float(csv16.perf[i])/1000)
        
        perf_list[16 + 0 ].append(float(csv1.perf[i+2688])/1000)
        perf_list[16 + 1 ].append(float(csv2.perf[i+2688])/1000)
        perf_list[16 + 2 ].append(float(csv3.perf[i+2688])/1000)
        perf_list[16 + 3 ].append(float(csv4.perf[i+2688])/1000)
        perf_list[16 + 4 ].append(float(csv5.perf[i+2688])/1000)
        perf_list[16 + 5 ].append(float(csv6.perf[i+2688])/1000)
        perf_list[16 + 6 ].append(float(csv7.perf[i+2688])/1000)
        perf_list[16 + 7 ].append(float(csv8.perf[i+2688])/1000)
        perf_list[16 + 8 ].append(float(csv9.perf[i+2688])/1000)
        perf_list[16 + 9 ].append(float(csv10.perf[i+2688])/1000)
        perf_list[16 + 10].append(float(csv11.perf[i+2688])/1000)
        perf_list[16 + 11].append(float(csv12.perf[i+2688])/1000)
        perf_list[16 + 12].append(float(csv13.perf[i+2688])/1000)
        perf_list[16 + 13].append(float(csv14.perf[i+2688])/1000)
        perf_list[16 + 14].append(float(csv15.perf[i+2688])/1000)
        perf_list[16 + 15].append(float(csv16.perf[i+2688])/1000)
        
        perf_list[32 + 0 ].append(float(csv1.perf[i+5376])/1000)
        perf_list[32 + 1 ].append(float(csv2.perf[i+5376])/1000)
        perf_list[32 + 2 ].append(float(csv3.perf[i+5376])/1000)
        perf_list[32 + 3 ].append(float(csv4.perf[i+5376])/1000)
        perf_list[32 + 4 ].append(float(csv5.perf[i+5376])/1000)
        perf_list[32 + 5 ].append(float(csv6.perf[i+5376])/1000)
        perf_list[32 + 6 ].append(float(csv7.perf[i+5376])/1000)
        perf_list[32 + 7 ].append(float(csv8.perf[i+5376])/1000)
        perf_list[32 + 8 ].append(float(csv9.perf[i+5376])/1000)
        perf_list[32 + 9 ].append(float(csv10.perf[i+5376])/1000)
        perf_list[32 + 10].append(float(csv11.perf[i+5376])/1000)
        perf_list[32 + 11].append(float(csv12.perf[i+5376])/1000)
        perf_list[32 + 12].append(float(csv13.perf[i+5376])/1000)
        perf_list[32 + 13].append(float(csv14.perf[i+5376])/1000)
        perf_list[32 + 14].append(float(csv15.perf[i+5376])/1000)
        perf_list[32 + 15].append(float(csv16.perf[i+5376])/1000)

        type1.append("vectorSparse-V1")
        type2.append("vectorSparse+SVC8")
        type3.append("WMMA-SpMM+SVC8")
        type4.append("SSpMM+SVC8")

print("valid data num: ", valid_num)
for i in range(4):
    for j in range(7):
        xlabel = sp[j]
        for k in range(96):
            NO_file = i*7*96+j*96+k
            if(int(csv4.nnz[NO_file])!=0):
                sparsity.append(xlabel)
########N = 512
VS1_volta_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[0],
    'lib': type1
})
VS8_volta_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[1],
    'lib': type2
})
WMMA_volta_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[2],
    'lib': type3
})
SSPMM_volta_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[3],
    'lib': type4
})
VS1_turing_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[4],
    'lib': type1
})
VS8_turing_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[5],
    'lib': type2
})
WMMA_turing_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[6],
    'lib': type3
})
SSPMM_turing_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[7],
    'lib': type4
})
VS1_ampere_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[8],
    'lib': type1
})
VS8_ampere_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[9],
    'lib': type2
})
WMMA_ampere_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[10],
    'lib': type3
})
SSPMM_ampere_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[11],
    'lib': type4
})
VS1_ada_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[12],
    'lib': type1
})
VS8_ada_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[13],
    'lib': type2
})
WMMA_ada_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[14],
    'lib': type3
})
SSPMM_ada_N512 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[15],
    'lib': type4
})
######N = 64
VS1_volta_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+0],
    'lib': type1
})
VS8_volta_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+1],
    'lib': type2
})
WMMA_volta_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+2],
    'lib': type3
})
SSPMM_volta_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+3],
    'lib': type4
})
VS1_turing_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+4],
    'lib': type1
})
VS8_turing_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+5],
    'lib': type2
})
WMMA_turing_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+6],
    'lib': type3
})
SSPMM_turing_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+7],
    'lib': type4
})
VS1_ampere_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+8],
    'lib': type1
})
VS8_ampere_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+9],
    'lib': type2
})
WMMA_ampere_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+10],
    'lib': type3
})
SSPMM_ampere_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+11],
    'lib': type4
})
VS1_ada_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+12],
    'lib': type1
})
VS8_ada_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+13],
    'lib': type2
})
WMMA_ada_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+14],
    'lib': type3
})
SSPMM_ada_N64 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[16+15],
    'lib': type4
})
##########N=4096
VS1_volta_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+0],
    'lib': type1
})
VS8_volta_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+1],
    'lib': type2
})
WMMA_volta_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+2],
    'lib': type3
})
SSPMM_volta_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+3],
    'lib': type4
})
VS1_turing_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+4],
    'lib': type1
})
VS8_turing_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+5],
    'lib': type2
})
WMMA_turing_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+6],
    'lib': type3
})
SSPMM_turing_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+7],
    'lib': type4
})
VS1_ampere_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+8],
    'lib': type1
})
VS8_ampere_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+9],
    'lib': type2
})
WMMA_ampere_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+10],
    'lib': type3
})
SSPMM_ampere_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+11],
    'lib': type4
})
VS1_ada_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+12],
    'lib': type1
})
VS8_ada_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+13],
    'lib': type2
})
WMMA_ada_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+14],
    'lib': type3
})
SSPMM_ada_N4096 = pd.DataFrame({
    'sparsity': sparsity,
    'perf': perf_list[32+15],
    'lib': type4
})

############N = 512
VS1_volta_N512_mean = VS1_volta_N512.groupby('sparsity')['perf'].mean()
VS8_volta_N512_mean = VS8_volta_N512.groupby('sparsity')['perf'].mean()
WMMA_volta_N512_mean = WMMA_volta_N512.groupby('sparsity')['perf'].mean()
SSPMM_volta_N512_mean = SSPMM_volta_N512.groupby('sparsity')['perf'].mean()
VS1_turing_N512_mean = VS1_turing_N512.groupby('sparsity')['perf'].mean()
VS8_turing_N512_mean = VS8_turing_N512.groupby('sparsity')['perf'].mean()
WMMA_turing_N512_mean = WMMA_turing_N512.groupby('sparsity')['perf'].mean()
SSPMM_turing_N512_mean = SSPMM_turing_N512.groupby('sparsity')['perf'].mean()
VS1_ampere_N512_mean = VS1_ampere_N512.groupby('sparsity')['perf'].mean()
VS8_ampere_N512_mean = VS8_ampere_N512.groupby('sparsity')['perf'].mean()
WMMA_ampere_N512_mean = WMMA_ampere_N512.groupby('sparsity')['perf'].mean()
SSPMM_ampere_N512_mean = SSPMM_ampere_N512.groupby('sparsity')['perf'].mean()
VS1_ada_N512_mean = VS1_ada_N512.groupby('sparsity')['perf'].mean()
VS8_ada_N512_mean = VS8_ada_N512.groupby('sparsity')['perf'].mean()
WMMA_ada_N512_mean = WMMA_ada_N512.groupby('sparsity')['perf'].mean()
SSPMM_ada_N512_mean = SSPMM_ada_N512.groupby('sparsity')['perf'].mean()
print("N = 512")
speedup_512_11 = np.mean(VS8_volta_N512_mean) / np.mean(VS1_volta_N512_mean)
speedup_512_12 = np.mean(WMMA_volta_N512_mean) / np.mean(VS8_volta_N512_mean)
speedup_512_13 = np.mean(SSPMM_volta_N512_mean) / np.mean(WMMA_volta_N512_mean)
speedup_512_1 = np.mean(SSPMM_volta_N512_mean) / np.mean(VS8_volta_N512_mean)
print("Volta:\t", speedup_512_11, speedup_512_12, speedup_512_13, speedup_512_1)
speedup_512_21 = np.mean(VS8_turing_N512_mean) / np.mean(VS1_turing_N512_mean)
speedup_512_22 = np.mean(WMMA_turing_N512_mean) / np.mean(VS8_turing_N512_mean)
speedup_512_23 = np.mean(SSPMM_turing_N512_mean) / np.mean(WMMA_turing_N512_mean)
speedup_512_2 = np.mean(SSPMM_turing_N512_mean) / np.mean(VS8_turing_N512_mean)
print("Turing:\t", speedup_512_21, speedup_512_22, speedup_512_23, speedup_512_2)
speedup_512_31 = np.mean(VS8_ampere_N512_mean) / np.mean(VS1_ampere_N512_mean)
speedup_512_32 = np.mean(WMMA_ampere_N512_mean) / np.mean(VS8_ampere_N512_mean)
speedup_512_33 = np.mean(SSPMM_ampere_N512_mean) / np.mean(WMMA_ampere_N512_mean)
speedup_512_3 = np.mean(SSPMM_ampere_N512_mean) / np.mean(VS8_ampere_N512_mean)
print("Ampere:\t", speedup_512_31, speedup_512_32, speedup_512_33, speedup_512_3)
speedup_512_41 = np.mean(VS8_ada_N512_mean) / np.mean(VS1_ada_N512_mean)
speedup_512_42 = np.mean(WMMA_ada_N512_mean) / np.mean(VS8_ada_N512_mean)
speedup_512_43 = np.mean(SSPMM_ada_N512_mean) / np.mean(WMMA_ada_N512_mean)
speedup_512_4 = np.mean(SSPMM_ada_N512_mean) / np.mean(VS8_ada_N512_mean)
print("Ada:\t", speedup_512_41, speedup_512_42, speedup_512_43, speedup_512_4)
############N = 64
VS1_volta_N64_mean = VS1_volta_N64.groupby('sparsity')['perf'].mean()
VS8_volta_N64_mean = VS8_volta_N64.groupby('sparsity')['perf'].mean()
WMMA_volta_N64_mean = WMMA_volta_N64.groupby('sparsity')['perf'].mean()
SSPMM_volta_N64_mean = SSPMM_volta_N64.groupby('sparsity')['perf'].mean()
VS1_turing_N64_mean = VS1_turing_N64.groupby('sparsity')['perf'].mean()
VS8_turing_N64_mean = VS8_turing_N64.groupby('sparsity')['perf'].mean()
WMMA_turing_N64_mean = WMMA_turing_N64.groupby('sparsity')['perf'].mean()
SSPMM_turing_N64_mean = SSPMM_turing_N64.groupby('sparsity')['perf'].mean()
VS1_ampere_N64_mean = VS1_ampere_N64.groupby('sparsity')['perf'].mean()
VS8_ampere_N64_mean = VS8_ampere_N64.groupby('sparsity')['perf'].mean()
WMMA_ampere_N64_mean = WMMA_ampere_N64.groupby('sparsity')['perf'].mean()
SSPMM_ampere_N64_mean = SSPMM_ampere_N64.groupby('sparsity')['perf'].mean()
VS1_ada_N64_mean = VS1_ada_N64.groupby('sparsity')['perf'].mean()
VS8_ada_N64_mean = VS8_ada_N64.groupby('sparsity')['perf'].mean()
WMMA_ada_N64_mean = WMMA_ada_N64.groupby('sparsity')['perf'].mean()
SSPMM_ada_N64_mean = SSPMM_ada_N64.groupby('sparsity')['perf'].mean()
print("N = 64")
speedup_64_11 = np.mean(VS8_volta_N64_mean) / np.mean(VS1_volta_N64_mean)
speedup_64_12 = np.mean(WMMA_volta_N64_mean) / np.mean(VS8_volta_N64_mean)
speedup_64_13 = np.mean(SSPMM_volta_N64_mean) / np.mean(WMMA_volta_N64_mean)
speedup_64_1 = np.mean(SSPMM_volta_N64_mean) / np.mean(VS8_volta_N64_mean)
print("Volta:\t", speedup_64_11, speedup_64_12, speedup_64_13, speedup_64_1)
speedup_64_21 = np.mean(VS8_turing_N64_mean) / np.mean(VS1_turing_N64_mean)
speedup_64_22 = np.mean(WMMA_turing_N64_mean) / np.mean(VS8_turing_N64_mean)
speedup_64_23 = np.mean(SSPMM_turing_N64_mean) / np.mean(WMMA_turing_N64_mean)
speedup_64_2 = np.mean(SSPMM_turing_N64_mean) / np.mean(VS8_turing_N64_mean)
print("Turing:\t", speedup_64_21, speedup_64_22, speedup_64_23, speedup_64_2)
speedup_64_31 = np.mean(VS8_ampere_N64_mean) / np.mean(VS1_ampere_N64_mean)
speedup_64_32 = np.mean(WMMA_ampere_N64_mean) / np.mean(VS8_ampere_N64_mean)
speedup_64_33 = np.mean(SSPMM_ampere_N64_mean) / np.mean(WMMA_ampere_N64_mean)
speedup_64_3 = np.mean(SSPMM_ampere_N64_mean) / np.mean(VS8_ampere_N64_mean)
print("Ampere:\t", speedup_64_31, speedup_64_32, speedup_64_33, speedup_64_3)
speedup_64_41 = np.mean(VS8_ada_N64_mean) / np.mean(VS1_ada_N64_mean)
speedup_64_42 = np.mean(WMMA_ada_N64_mean) / np.mean(VS8_ada_N64_mean)
speedup_64_43 = np.mean(SSPMM_ada_N64_mean) / np.mean(WMMA_ada_N64_mean)
speedup_64_4 = np.mean(SSPMM_ada_N64_mean) / np.mean(VS8_ada_N64_mean)
print("Ada:\t", speedup_64_41, speedup_64_42, speedup_64_43, speedup_64_4)
############N = 4096
VS1_volta_N4096_mean = VS1_volta_N4096.groupby('sparsity')['perf'].mean()
VS8_volta_N4096_mean = VS8_volta_N4096.groupby('sparsity')['perf'].mean()
WMMA_volta_N4096_mean = WMMA_volta_N4096.groupby('sparsity')['perf'].mean()
SSPMM_volta_N4096_mean = SSPMM_volta_N4096.groupby('sparsity')['perf'].mean()
VS1_turing_N4096_mean = VS1_turing_N4096.groupby('sparsity')['perf'].mean()
VS8_turing_N4096_mean = VS8_turing_N4096.groupby('sparsity')['perf'].mean()
WMMA_turing_N4096_mean = WMMA_turing_N4096.groupby('sparsity')['perf'].mean()
SSPMM_turing_N4096_mean = SSPMM_turing_N4096.groupby('sparsity')['perf'].mean()
VS1_ampere_N4096_mean = VS1_ampere_N4096.groupby('sparsity')['perf'].mean()
VS8_ampere_N4096_mean = VS8_ampere_N4096.groupby('sparsity')['perf'].mean()
WMMA_ampere_N4096_mean = WMMA_ampere_N4096.groupby('sparsity')['perf'].mean()
SSPMM_ampere_N4096_mean = SSPMM_ampere_N4096.groupby('sparsity')['perf'].mean()
VS1_ada_N4096_mean = VS1_ada_N4096.groupby('sparsity')['perf'].mean()
VS8_ada_N4096_mean = VS8_ada_N4096.groupby('sparsity')['perf'].mean()
WMMA_ada_N4096_mean = WMMA_ada_N4096.groupby('sparsity')['perf'].mean()
SSPMM_ada_N4096_mean = SSPMM_ada_N4096.groupby('sparsity')['perf'].mean()
print("N = 4096")
speedup_4096_11 = np.mean(VS8_volta_N4096_mean) / np.mean(VS1_volta_N4096_mean)
speedup_4096_12 = np.mean(WMMA_volta_N4096_mean) / np.mean(VS8_volta_N4096_mean)
speedup_4096_13 = np.mean(SSPMM_volta_N4096_mean) / np.mean(WMMA_volta_N4096_mean)
speedup_4096_1 = np.mean(SSPMM_volta_N4096_mean) / np.mean(VS8_volta_N4096_mean)
print("Volta:\t", speedup_4096_11, speedup_4096_12, speedup_4096_13, speedup_4096_1)
speedup_4096_21 = np.mean(VS8_turing_N4096_mean) / np.mean(VS1_turing_N4096_mean)
speedup_4096_22 = np.mean(WMMA_turing_N4096_mean) / np.mean(VS8_turing_N4096_mean)
speedup_4096_23 = np.mean(SSPMM_turing_N4096_mean) / np.mean(WMMA_turing_N4096_mean)
speedup_4096_2 = np.mean(SSPMM_turing_N4096_mean) / np.mean(VS8_turing_N4096_mean)
print("Turing:\t", speedup_4096_21, speedup_4096_22, speedup_4096_23, speedup_4096_2)
speedup_4096_31 = np.mean(VS8_ampere_N4096_mean) / np.mean(VS1_ampere_N4096_mean)
speedup_4096_32 = np.mean(WMMA_ampere_N4096_mean) / np.mean(VS8_ampere_N4096_mean)
speedup_4096_33 = np.mean(SSPMM_ampere_N4096_mean) / np.mean(WMMA_ampere_N4096_mean)
speedup_4096_3 = np.mean(SSPMM_ampere_N4096_mean) / np.mean(VS8_ampere_N4096_mean)
print("Ampere:\t", speedup_4096_31, speedup_4096_32, speedup_4096_33, speedup_4096_3)
speedup_4096_41 = np.mean(VS8_ada_N4096_mean) / np.mean(VS1_ada_N4096_mean)
speedup_4096_42 = np.mean(WMMA_ada_N4096_mean) / np.mean(VS8_ada_N4096_mean)
speedup_4096_43 = np.mean(SSPMM_ada_N4096_mean) / np.mean(WMMA_ada_N4096_mean)
speedup_4096_4 = np.mean(SSPMM_ada_N4096_mean) / np.mean(VS8_ada_N4096_mean)
print("Ada:\t", speedup_4096_41, speedup_4096_42, speedup_4096_43, speedup_4096_4)
# total_v8volta_mean = np.mean(v8volta_mean)
# vsvolta_mean = df_vs_volta.groupby('sparsity')['perf'].mean()
# total_vsvolta_mean = np.mean(vsvolta_mean)

# t8turing_mean = df_t8_turing.groupby('sparsity')['perf'].mean()
# total_t8turing_mean = np.mean(t8turing_mean)
# v8turing_mean = df_v8_turing.groupby('sparsity')['perf'].mean()
# total_v8turing_mean = np.mean(v8turing_mean)
# vsturing_mean = df_vs_turing.groupby('sparsity')['perf'].mean()
# total_vsturing_mean = np.mean(vsturing_mean)
# t8ampere_mean = df_t8_ampere.groupby('sparsity')['perf'].mean()
# total_t8ampere_mean = np.mean(t8ampere_mean)
# v8ampere_mean = df_v8_ampere.groupby('sparsity')['perf'].mean()
# total_v8ampere_mean = np.mean(v8ampere_mean)
# vsampere_mean = df_vs_ampere.groupby('sparsity')['perf'].mean()
# total_vsampere_mean = np.mean(vsampere_mean)
# t8ada_mean = df_t8_ada.groupby('sparsity')['perf'].mean()
# total_t8ada_mean = np.mean(t8ada_mean)
# v8ada_mean = df_v8_ada.groupby('sparsity')['perf'].mean()
# total_v8ada_mean = np.mean(v8ada_mean)
# vsada_mean = df_vs_ada.groupby('sparsity')['perf'].mean()
# total_vsada_mean = np.mean(vsada_mean)
# # for i in range(7):
# #     print("Turing-V8: ", sp[i], "  ", t8turing_mean[i], v8turing_mean[i])
# #     print("Ampere-V8: ", sp[i], "  ", t8ampere_mean[i], v8ampere_mean[i])
# #     print("Ada-V8: ", sp[i], "  ", t8ada_mean[i], v8ada_mean[i], vsada_mean[i])
# print("volta:")
# print(total_v8volta_mean, total_vsvolta_mean)
# print(total_v8volta_mean/total_vsvolta_mean)

# print("turing:")
# print(total_t8turing_mean, total_v8turing_mean, total_vsturing_mean)
# print(total_t8turing_mean/total_vsturing_mean, total_v8turing_mean/total_vsturing_mean)
# print("ampere:")
# print(total_t8ampere_mean, total_v8ampere_mean, total_vsampere_mean)
# print(total_t8ampere_mean/total_vsampere_mean, total_v8ampere_mean/total_vsampere_mean)
# print("ada:")
# print(total_t8ada_mean, total_v8ada_mean, total_vsada_mean)
# print(total_t8ada_mean/total_vsada_mean, total_v8ada_mean/total_vsada_mean)

x = np.arange(len(sp))  # the label locations
width = 0.22  # the width of the bars

fig,ax = plt.subplots(3, 4, figsize=(32, 10.3)) 
plt.subplot(3,4,1)
plt.bar(x - 2*width, VS1_volta_N512_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_volta_N512_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse-V8')
plt.bar(x          , WMMA_volta_N512_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM-V8')
plt.bar(x +   width, SSPMM_volta_N512_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM-V8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_volta_N512_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(" ",fontsize=28)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,6)
plt.yticks(np.arange(0, 7, 1))
# plt.title('Scores by group and gender',fontsize=28)
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(sp,rotation=0, fontsize = 24)
ax[0,0].set_title('V100 (Volta Architecture)',fontsize=24)
ax[0,0].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,2)
plt.bar(x - 2*width, VS1_turing_N512_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_turing_N512_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse-V8')
plt.bar(x          , WMMA_turing_N512_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM-V8')
plt.bar(x +   width, SSPMM_turing_N512_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM-V8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_turing_N512_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,6)
plt.yticks(np.arange(0, 7, 1))
# plt.title('Scores by group and gender',fontsize=28)
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(sp,rotation=0, fontsize = 24)
ax[0,1].set_title('2080Ti (Turing Architecture)',fontsize=24)
ax[0,1].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,3)
plt.bar(x - 2*width, VS1_ampere_N512_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ampere_N512_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_ampere_N512_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_ampere_N512_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ampere_N512_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,9)
plt.yticks(np.arange(0, 10, 2))
# plt.title('Scores by group and gender',fontsize=28)
ax[0,2].set_xticks(x)
ax[0,2].set_xticklabels(sp,rotation=0, fontsize = 24)
ax[0,2].set_title('A100 (Ampere Architecture)',fontsize=24)
ax[0,2].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,4)
plt.bar(x - 2*width, VS1_ada_N512_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ada_N512_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_ada_N512_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_ada_N512_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ada_N512_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,15)
plt.yticks(np.arange(0, 18, 3))
# plt.title('Scores by group and gender',fontsize=28)
ax[0,3].set_xticks(x)
ax[0,3].set_xticklabels(sp,rotation=0, fontsize = 24)
ax[0,3].set_title('4090 (Ada Architecture)',fontsize=24)
ax[0,3].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
############################################N=64#######################################
plt.subplot(3,4,5)
plt.bar(x - 2*width, VS1_volta_N64_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_volta_N64_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_volta_N64_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_volta_N64_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_volta_N64_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel("Performance(TFLOPS)",fontsize=28)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,2.5)
plt.yticks(np.arange(0, 3, 0.5))
# plt.title('Scores by group and gender',fontsize=28)
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[1,0].set_title('V100 (Volta Architecture)',fontsize=24)
ax[1,0].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,6)
plt.bar(x - 2*width, VS1_turing_N64_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_turing_N64_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_turing_N64_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_turing_N64_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_turing_N64_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,2.5)
plt.yticks(np.arange(0, 3, 0.5))
# plt.title('Scores by group and gender',fontsize=28)
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[1,1].set_title('2080Ti (Turing Architecture)',fontsize=24)
ax[1,1].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,7)
plt.bar(x - 2*width, VS1_ampere_N64_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ampere_N64_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_ampere_N64_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_ampere_N64_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ampere_N64_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,2.5)
plt.yticks(np.arange(0, 3, 0.5))
# plt.title('Scores by group and gender',fontsize=28)
ax[1,2].set_xticks(x)
ax[1,2].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[1,2].set_title('A100 (Ampere Architecture)',fontsize=24)
ax[1,2].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,8)
plt.bar(x - 2*width, VS1_ada_N64_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ada_N64_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_ada_N64_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_ada_N64_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ada_N64_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,3)
plt.yticks(np.arange(0, 3.5, 0.5))
# plt.title('Scores by group and gender',fontsize=28)
ax[1,3].set_xticks(x)
ax[1,3].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[1,3].set_title('4090 (Ada Architecture)',fontsize=24)
ax[1,3].set_xlabel(" ",fontsize=0)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)
############################################N=4096#######################################
plt.subplot(3,4,9)
plt.bar(x - 2*width, VS1_volta_N4096_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_volta_N4096_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_volta_N4096_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_volta_N4096_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_volta_N4096_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(" ",fontsize=28)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,10)
plt.yticks(np.arange(0, 11, 2))
ax[2,0].set_xticks(x)
ax[2,0].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[2,0].set_title('V100 (Volta Architecture)',fontsize=24)
ax[2,0].set_xlabel("Sparsity",fontsize=28)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,10)
plt.bar(x - 2*width, VS1_turing_N4096_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_turing_N4096_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_turing_N4096_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_turing_N4096_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_turing_N4096_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,7)
plt.yticks(np.arange(0, 8, 1))
# plt.title('Scores by group and gender',fontsize=28)
ax[2,1].set_xticks(x)
ax[2,1].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[2,1].set_title('2080Ti (Turing Architecture)',fontsize=24)
ax[2,1].set_xlabel(" ",fontsize=45)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

plt.subplot(3,4,11)
plt.bar(x - 2*width, VS1_ampere_N4096_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ampere_N4096_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+SVC8')
plt.bar(x          , WMMA_ampere_N4096_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM+SVC8')
plt.bar(x +   width, SSPMM_ampere_N4096_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM+SVC8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ampere_N4096_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,16)
plt.yticks(np.arange(0, 18, 3))
# plt.title('Scores by group and gender',fontsize=28)
ax[2,2].set_xticks(x)
ax[2,2].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[2,2].set_title('A100 (Ampere Architecture)',fontsize=24)
ax[2,2].set_xlabel(" ",fontsize=45)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

g = plt.subplot(3,4,12)
plt.bar(x - 2*width, VS1_ada_N4096_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse-V1')
plt.bar(x -   width, VS8_ada_N4096_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse-V8')
plt.bar(x          , WMMA_ada_N4096_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='WMMA-SpMM-V8')
plt.bar(x +   width, SSPMM_ada_N4096_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5,label='SSpMM-V8')
# for a,b in zip(x,VS1_volta_N512_mean): ##label position
#     plt.text(a-2*width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,VS8_volta_N512_mean): ##label position
#     plt.text(a - width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
# for a,b in zip(x,WMMA_volta_N512_mean): ##label position
#     plt.text(a        ,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)
for a,b in zip(x,SSPMM_ada_N4096_mean): ##label position
    plt.text(a + width,b+0,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=23)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(' ',fontsize=0)
plt.xlabel(" ",fontsize=0)
plt.ylim(0,36)
plt.yticks(np.arange(0, 40, 6))
# plt.title('Scores by group and gender',fontsize=28)
ax[2,3].set_xticks(x)
ax[2,3].set_xticklabels(sp,rotation=0, fontsize = 24)
# ax[2,3].set_title('4090 (Ada Architecture)',fontsize=24)
ax[2,3].set_xlabel("Sparsity",fontsize=28)
plt.tick_params(labelsize=24)
plt.grid(c='grey',alpha=0.9,linestyle='--')
# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
plt.legend([], [], frameon=False)

handles, labels = g.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=28)
fig.text(0.195, 0.93, 'N = 512', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.195, 0.62, 'N = 64', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.195, 0.31, 'N = 4096', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.44, 0.93, 'N = 512', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.44, 0.62, 'N = 64', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.44, 0.31, 'N = 4096', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.685, 0.93, 'N = 512', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.685, 0.62, 'N = 64', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.685, 0.31, 'N = 4096', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.93, 0.93, 'N = 512', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.93, 0.62, 'N = 64', va='center', rotation='horizontal', fontsize = 30)
fig.text(0.93, 0.31, 'N = 4096', va='center', rotation='horizontal', fontsize = 30)

fig.tight_layout()
plt.savefig('exp2.pdf',dpi=300)
# plt.show()

