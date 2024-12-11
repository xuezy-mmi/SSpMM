import re
import six
import csv

#C:/MATT/CUDA/Magicube-master/plot/examples
filename = "./pytorch_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

reader = open(filename, 'r')
runtime = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime.append(m.group(1))

#writer = open('0_pytorch_n2n.txt', 'w')
#for r in runtime:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "./vectorSparse_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

reader = open(filename, 'r')
runtime1 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime1.append(m.group(1))

#writer = open('0_vectorSparse_n2n.txt', 'w')
#for r in runtime1:
#    writer.write(str(r) + '\n')
#writer.close()


filename = "./RoSpMM_n2n.txt"
pattern = r"runtime (\d+(\.\d+)?) milliseconds"

reader = open(filename, 'r')
runtime2 = []

while True:
    line = reader.readline()
    if len(line) == 0:
        break
    line = line.rstrip()
    m = re.search(pattern, line)
    if m:
        runtime2.append(m.group(1))

#3writer = open('0_magicube_n2n.txt', 'w')
#3for r in runtime2:
#3    writer.write(str(r) + '\n')
#3writer.close()


header = ['S0.9,Seq_l=4096,num_h=4', 'algs', 'Latency(ms)']                                     
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_a = 0.0
vectorsparse_a = 0.0
RoSpMM_a = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i]))
            cudnn_a += float(runtime[i])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i]))
            vectorsparse_a += float(runtime1[i])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i]))
            RoSpMM_a += float(runtime2[i])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+8]))
            cudnn_a += float(runtime[i+8])
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+8]))
            vectorsparse_a += float(runtime1[i+8])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+8]))
            RoSpMM_a += float(runtime2[i+8])
with open('n2n_a.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_a0 = cudnn_a / RoSpMM_a
speedup_a1 = vectorsparse_a / RoSpMM_a
print("speedup(cudnn/RoSpMM): ", speedup_a0)
print("speedup(vs/RoSpMM)   : ", speedup_a1)



header = ['S0.9,Seq_l=4096,num_h=8', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_b = 0.0
vectorsparse_b = 0.0
RoSpMM_b = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+16]))
            cudnn_b += float(runtime[i+16])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+16]))
            vectorsparse_b += float(runtime1[i+16])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+16]))
            RoSpMM_b += float(runtime2[i+16])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+24]))
            cudnn_b += float(runtime[i+24])
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+24]))
            vectorsparse_b += float(runtime1[i+24])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+24]))
            RoSpMM_b += float(runtime2[i+24])
with open('n2n_b.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_b0 = cudnn_b / RoSpMM_b
speedup_b1 = vectorsparse_b / RoSpMM_b
print("speedup(cudnn/RoSpMM): ", speedup_b0)
print("speedup(vs/RoSpMM)   : ", speedup_b1)

header = ['S0.9,Seq_l=8192,num_h=4', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_c = 0.0
vectorsparse_c = 0.0
RoSpMM_c = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+32]))
            cudnn_c += float(runtime[i+32])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+32]))
            vectorsparse_c += float(runtime1[i+32])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+32]))
            RoSpMM_c += float(runtime2[i+32])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(None)
            cudnn_c += float(runtime[i+32])*4
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+40]))
            vectorsparse_c += float(runtime1[i+40])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+40]))
            RoSpMM_c += float(runtime2[i+40])
with open('n2n_c.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_c0 = cudnn_c / RoSpMM_c
speedup_c1 = vectorsparse_c / RoSpMM_c
print("speedup(cudnn/RoSpMM): ", speedup_c0)
print("speedup(vs/RoSpMM)   : ", speedup_c1)

header = ['S0.9,Seq_l=8192,num_h=8', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_d = 0.0
vectorsparse_d = 0.0
RoSpMM_d = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+40]))
            cudnn_d += float(runtime[i+40])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+48]))
            vectorsparse_d += float(runtime1[i+48])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+48]))
            RoSpMM_d += float(runtime2[i+48])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(None)
            cudnn_d += float(runtime[i+40])*4
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+56]))
            vectorsparse_d += float(runtime1[i+56])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+56]))
            RoSpMM_d += float(runtime2[i+56])
with open('n2n_d.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)


print(header[0])
speedup_d0 = cudnn_d / RoSpMM_d
speedup_d1 = vectorsparse_d / RoSpMM_d
print("speedup(cudnn/RoSpMM): ", speedup_d0)
print("speedup(vs/RoSpMM)   : ", speedup_d1)


header = ['S0.95,Seq_l=4096,num_h=4', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_e = 0.0
vectorsparse_e = 0.0
RoSpMM_e = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i]))
            cudnn_e += float(runtime[i])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+64]))
            vectorsparse_e += float(runtime1[i+64])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+64]))
            RoSpMM_e += float(runtime2[i+64])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+8]))
            cudnn_e += float(runtime[i+8])
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+72]))
            vectorsparse_e += float(runtime1[i+72])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+72]))
            RoSpMM_e += float(runtime2[i+72])
with open('n2n_e.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_e0 = cudnn_e / RoSpMM_e
speedup_e1 = vectorsparse_e / RoSpMM_e
print("speedup(cudnn/RoSpMM): ", speedup_e0)
print("speedup(vs/RoSpMM)   : ", speedup_e1)


header = ['S0.95,Seq_l=4096,num_h=8', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_f = 0.0
vectorsparse_f = 0.0
RoSpMM_f = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+16]))
            cudnn_f += float(runtime[i+16])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+80]))
            vectorsparse_f += float(runtime1[i+80])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+80]))
            RoSpMM_f += float(runtime2[i+80])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+24]))
            cudnn_f += float(runtime[i+24])
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+88]))
            vectorsparse_f += float(runtime1[i+88])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+88]))
            RoSpMM_f += float(runtime2[i+88])
with open('n2n_f.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_f0 = cudnn_f / RoSpMM_f
speedup_f1 = vectorsparse_f / RoSpMM_f
print("speedup(cudnn/RoSpMM): ",speedup_f0)
print("speedup(vs/RoSpMM)   : ", speedup_f1)

header = ['S0.95,Seq_l=8192,num_h=4', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_g = 0.0
vectorsparse_g = 0.0
RoSpMM_g = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+32]))
            cudnn_g += float(runtime[i+32])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+96]))
            vectorsparse_g += float(runtime1[i+96])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+96]))
            RoSpMM_g += float(runtime2[i+96])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(None)
            cudnn_g += float(runtime[i+32])*4
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+104]))
            vectorsparse_g += float(runtime1[i+104])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+104]))
            RoSpMM_g += float(runtime2[i+104])
with open('n2n_g.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_g0 = cudnn_g / RoSpMM_g
speedup_g1 = vectorsparse_g / RoSpMM_g
print("speedup(cudnn/RoSpMM): ", speedup_g0)
print("speedup(vs/RoSpMM)   : ", speedup_g1)

header = ['S0.95,Seq_l=8192,num_h=8', 'algs', 'Latency(ms)']
m = 8 
data_list = [[] * 3 for i in range(m*6)]
cudnn_h = 0.0
vectorsparse_h = 0.0
RoSpMM_h = 0.0
# print(len(data_list))
for ll in range(6):
    if ll < 3:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=2')
    else:
        for i in range(m):
            data_list[i + ll*m].append('batchsize=8')


    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append('Pytorch-fp16')
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append('vectorSparse')
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append('RoSpMM')

    if ll == 0:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime[i+40]))
            cudnn_h += float(runtime[i+40])
    if ll == 1:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+112]))
            vectorsparse_h += float(runtime1[i+112])
    if ll == 2:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+112]))
            RoSpMM_h += float(runtime2[i+112])
    if ll == 3:
        for i in range(m):
            data_list[i + ll*m].append(None)
            cudnn_h += float(runtime[i+40])*4
    if ll == 4:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime1[i+120]))
            vectorsparse_h += float(runtime1[i+120])
    if ll == 5:
        for i in range(m):
            data_list[i + ll*m].append(float(runtime2[i+120]))
            RoSpMM_h += float(runtime2[i+120])
with open('n2n_h.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data_list)

print(header[0])
speedup_h0 = cudnn_h / RoSpMM_h
speedup_h1 = vectorsparse_h / RoSpMM_h
print("speedup(cudnn/RoSpMM): ", speedup_h0)
print("speedup(vs/RoSpMM)   : ", speedup_h1)


print("\nseq_length=4096\n")
print("speedup(cudnn/RoSpMM): ", (speedup_a0+speedup_b0+speedup_e0+speedup_f0)/4)
print("speedup(vs/RoSpMM)   : ", (speedup_a1+speedup_b1+speedup_e1+speedup_f1)/4)


print("\n0.90\n")
print("speedup(cudnn/RoSpMM): ", (speedup_a0+speedup_b0+speedup_c0+speedup_d0)/4)
print("speedup(vs/RoSpMM)   : ", (speedup_a1+speedup_b1+speedup_c1+speedup_d1)/4)


print("\n0.95\n")
print("speedup(cudnn/RoSpMM): ", (speedup_e0+speedup_f0+speedup_g0+speedup_h0)/4)
print("speedup(vs/RoSpMM)   : ", (speedup_e1+speedup_f1+speedup_g1+speedup_h1)/4)


print("\ntotal\n")
print("speedup(cudnn/RoSpMM): ", (speedup_a0+speedup_b0+speedup_c0+speedup_d0+speedup_e0+speedup_f0+speedup_g0+speedup_h0)/8)
print("speedup(vs/RoSpMM)   : ", (speedup_a1+speedup_b1+speedup_c1+speedup_d1+speedup_e1+speedup_f1+speedup_g1+speedup_h1)/8)
