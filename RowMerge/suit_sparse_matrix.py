from scipy.io import mmread
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math
import csv
import os
import random
import re
import sparse_matrix_format

# class CVSE:
#     def __init__(self, ROW, COL, NNZ, colInd_list_array, vec_len):

#         self.M = ROW#number of rows
#         self.K = COL#number of cols
#         self.nnz = NNZ#number of non-zero element
#         self.vec_len = vec_len#vector length
#         self.num_vec_row = math.ceil(self.M / vec_len)#number of vector-row
#         self.tile_num = [0 for i in range(num_vec_row)]
#         num_vec = 0
#         vec_row_ptr = [0]
#         vec_colInd_list_array = []#format:[[],[],...,[],[]]

#         for i in range(self.num_vec_row):
#             vec_colInd = []
#             for j in range(self.vec_len):
#                 vec_colInd = list(set(vec_colInd + colInd_list_array[self.vec_len*i+j]))###this vec_row's colInd
#             vec_colInd_list_array.append(vec_colInd)
#             num_vec = num_vec + len(vec_colInd)
#             vec_row_ptr.append(num_vec)
#             self.tile_num[i] = math.ceil(len(vec_colInd) / 32)
#         self.vec_colInd_list_array = vec_colInd_list_array#format:[[],[],...,[],[]]
#         self.num_vec = num_vec#number of non-zero vector
#         self.vec_row_ptr = vec_row_ptr#ptr of vec_row

#         vec_colInd_list = []
#         for i in range(self.num_vec_row):
#             for j in range(len(self.vec_colInd_list_array[i])):
#                 vec_colInd_list.append(self.vec_colInd_list_array[i][j])
#         self.vec_colInd_list = vec_colInd_list
#     def return_CVSE_storage(self):
#         index_type = 4#32/8=4
#         value_type = 2#16/8=2
#         storage = self.nnz * value_type + self.num_vec * index_type + (self.num_vec_row + 1) * index_type
#         return storage

#     def return_TCF_storage(self):
#         index_type = 4#32/8=4
#         value_type = 2#16/8=2
#         storage = nnz * value_type
#         return storage

#     def return_SRBCSR_storage(self):
#         index_type = 4#32/8=4
#         value_type = 2#16/8=2
#         storage = self.nnz * value_type + (self.num_vec_row + 1) * index_type
#         for i in range(self.num_vec_row):
#             storage += math.ceil(tile_num[i]/32) * 32 * index_type
#         return storage
    
#     def CVSE_out(self, new_dir):
#         # new_path = dir + self.filename
#         # print(new_dir)
#         with open(new_dir, "w") as f:
#             f.write(str(self.num_vec_row))
#             f.write(", ")
#             f.write(str(self.K))
#             f.write(", ")
#             f.write(str(self.nnz))
#             f.write("\n")
#             for i in range(len(self.vec_row_ptr)):
#                 f.write(str(self.vec_row_ptr[i]))
#                 if(i == len(self.vec_row_ptr)-1):
#                     break
#                 f.write(" ")
#             f.write("\n")
#             for j in range(self.num_vec):
#                 # if(self.vec_colInd_list[j] >= self.K):
#                 #     print("\n\n\n\n\n\n\nerror\n\n\n\n\n\n")
#                 #     break
#                 f.write(str(self.vec_colInd_list[j]))
#                 if(j == self.num_vec - 1 ):
#                     break
#                 f.write(" ")

        
class suite_matrix:
    def __init__(self, path):
        matrix = mmread(path)
        coo = coo_matrix(matrix)
        num_row, num_col = coo.shape
        # NNZ = coo.nnz
        # row_indices = coo.row
        # col_indices = coo.col
        # # value = coo.data
        self.path = path
        self.coo = coo
        self.M = num_row#number of rows
        self.K = num_col#number of cols
        self.nnz = coo.nnz#number of non-zero element
        row_indices = []
        col_indices = []
        for i in range(self.nnz):
            row_indices.append(coo.row[i]-1)
            col_indices.append(coo.col[i]-1)
        self.row_indices = row_indices
        self.col_indices = col_indices
        # self.vec_len = vec_len#vector length
        # self.num_vec_row = math.ceil(self.M / vec_len)#number of vector-row
    def coo2csr(self, gen = 0):
        csr = self.coo.tocsr()
        Row_Ptr = csr.indptr
        Col_Indices = csr.indices
        new_filename = self.path.replace(".mtx", "_csr.mtx")
        # print(new_filename)
        if(gen == 1):
            with open(new_filename, "w") as f:
                f.write(str(self.M))
                f.write(", ")
                f.write(str(self.K))
                f.write(", ")
                f.write(str(self.nnz))
                f.write("\n")
                for i in range(len(Row_Ptr)):
                    f.write(str(Row_Ptr[i]))
                    if(i == len(Row_Ptr)-1):
                        break
                    f.write(" ")
                f.write("\n")
                for j in range(len(Col_Indices)):
                    # if(self.vec_colInd_list[j] >= self.K):
                    #     print("\n\n\n\n\n\n\nerror\n\n\n\n\n\n")
                    #     break
                    f.write(str(Col_Indices[j]))
                    if(j == len(Col_Indices)-1 ):
                        break
                    f.write(" ")
            print("Gnerate _csr.mtx File: ", new_filename)
        return Row_Ptr, Col_Indices
    def colInd_list_array(self):
        Row_Ptr, Col_Indices = self.coo2csr()
        colInd_list_array = []
        for i in range(self.M):
            this_row_colind = []
            for j in Col_Indices[Row_Ptr[i]:Row_Ptr[i+1]]:
                # origin_matrix[i][j] = 1
                this_row_colind.append(j)
            colInd_list_array.append(this_row_colind)
        return colInd_list_array
    def sparsity(self):
        sparsity_ratio = 1.0 - self.nnz / self.M / self.K
        return sparsity_ratio
    def origin_matrix(self):#return matrix[ [], [], [] ]
        ROW = self.M
        csr = self.coo.tocsr()
        ROW_PTR = csr.indptr
        COL_IND = csr.indices
        origin_matrix = [[0 for i in range(self.K)] for j in range(self.M)]
        # array = []
        for i in range(ROW):
            # this_row_colind = []
            for j in COL_IND[ROW_PTR[i]:ROW_PTR[i+1]]:
                origin_matrix[i][j] = 1
                # this_row_colind.append(j)
            # array.append(this_row_colind)
        return origin_matrix
    
def return_all_zero_vec(vec_len, ROW, COL, origin_matrix):
    num_vec_row = math.ceil(ROW / vec_len)
    cnt = 0
    for i in range(num_vec_row):
        for j in range(COL):
            flag = 0
            for k in range(vec_len):
                if(i*vec_len+k < ROW):
                    flag += origin_matrix[i*vec_len+k][j]
            if(flag == 0):
                cnt += 1
    return cnt

def generate_suitesparse_CVSE_file(src_dir, dest_dir, paths_txt, vec_len):
    # paths = src_dir + paths_txt
    # file_name = "./csv_data/RowMerge_suit_V" + str(vec_len) + ".csv"
    # fcsv = open(file_name, 'a')
    # writer = csv.writer(fcsv)
    
    with open(paths_txt, 'r') as file:
        print("Begin to gen CSR-file and CVSE-file, vecLength=", vec_len)
        lines = [line.rstrip('\n') for line in file]
        file_num = len(lines)
        print("Number of file is ",file_num)
        for i in range(file_num):
            # filename = file.readline().split()[0]
            source_file_name = src_dir + lines[i]
            dest_file_name = dest_dir + lines[i]
            dest_file_name = dest_file_name.replace(".mtx", ".smtx")
            print("source: ", source_file_name)
            print("dest:   ", dest_file_name)
            suite_M = suite_matrix(source_file_name)
            original_sparsity = 1.0 - suite_M.nnz / suite_M.M / suite_M.K
            print(suite_M.M, suite_M.K, suite_M.nnz, original_sparsity)
            suite_M.coo2csr(1)# write csr file
            # origin_matrix = suite_M.origin_matrix()
            colInd_list_array = suite_M.colInd_list_array()
            # all_zero_vec_cnt = return_all_zero_vec(vec_len, suite_M.M, suite_M.K, origin_matrix)
            # vec_cnt = math.ceil(suite_M.M / vec_len) * suite_M.K - all_zero_vec_cnt
            # if(vec_cnt == 0):
                # new_sparsity = 1.0
            # else:
                # new_sparsity = 1.0 - suite_M.nnz/(vec_cnt * vec_len)
            cvse_M = sparse_matrix_format.CVSE(suite_M.M, suite_M.K, suite_M.nnz, colInd_list_array, vec_len)
            # CSR_storage = cvse_M.return_CSR_storage()
            # CVSE_storage = cvse_M.return_CVSE_storage()
            # TCF_storage  = cvse_M.return_TCF_storage()
            # SR_BCSR_storage = cvse_M.return_SRBCSR_storage()
            cvse_M.CVSE_out(dest_file_name)
            # this_csv_row = [str(i), str(suite_M.M), str(suite_M.K), str(suite_M.nnz), str(new_sparsity), str(all_zero_vec_cnt), str(CSR_storage), str(CVSE_storage), str(TCF_storage), str(SR_BCSR_storage)]
            # writer.writerow(this_csv_row)
            print("File-", i, " Finished")
            # print(suite_M.M, suite_M.K, suite_M.nnz, new_sparsity, all_zero_vec_cnt)
            # print(CSR_storage, CVSE_storage, TCF_storage, SR_BCSR_storage)

class suit_csr:
    def __init__(self, path):
        with open(path) as f:#####read from .mtx and init
            data = f.readline()
            rowPtr_s = f.readline()
            colInd_s = f.readline()
            M, N, nnz = data.split(', ')
            rowPtr = rowPtr_s.split()
            colInd = colInd_s.split()
            rowPtr_len = len(rowPtr)
            colInd_len = len(colInd)
        f.close()
        M = int(M)
        N = int(N)
        nnz = int(nnz)
        print(path, " is Read Over")
        print(M, N, nnz)
        for i in range(rowPtr_len):
            rowPtr[i] = int(rowPtr[i])
        for i in range(colInd_len):
            colInd[i] = int(colInd[i])
        colInd_list_array = []
        origin_matrix = [[0 for i in range(N)] for j in range(M)]
        for i in range(M):
            this_row_colind = []
            for j in colInd[rowPtr[i]:rowPtr[i+1]]:
                this_row_colind.append(j)
                origin_matrix[i][j] = 1
            colInd_list_array.append(this_row_colind)
                
        self.row = M
        self.col = N
        self.nnz = nnz
        self.rowPtr = rowPtr##########array of CSR's rowPtr
        self.colInd = colInd##########array of CSR's col Index########COO's col Index
        self.origin_matrix = origin_matrix
        self.colInd_list_array = colInd_list_array
        print("CSR Init Done")
    def CSR_storage(self):
        index_type = 4#32/8=4
        value_type = 2#16/8=2
        storage = self.nnz * (value_type + index_type) + (self.row + 1) * index_type
        return storage
    def return_all_zero_vec(self, vec_len):
        num_vec_row = math.ceil(self.row / vec_len)
        cnt = 0
        for i in range(num_vec_row):
            for j in range(self.col):
                flag = 0
                for k in range(vec_len):
                    if(i*vec_len+k < self.row):
                        flag += origin_matrix[i*vec_len+k][j]
                if(flag == 0):
                    cnt += 1
        return cnt

class suit_cvse:
    def __init__(self, path, vec_len):
        with open(path) as f:#####read from .mtx and init
            data = f.readline()
            rowPtr_s = f.readline()
            colInd_s = f.readline()
            M, N, nnz = data.split(', ')
            rowPtr = rowPtr_s.split()
            colInd = colInd_s.split()
            rowPtr_len = len(rowPtr)#M+1
            colInd_len = len(colInd)
        f.close()
        num_vec_row = int(M)
        N = int(N)
        nnz = int(nnz)
        print(path, " is Read Over")
        print(num_vec_row, N, nnz)
        num_vec = colInd_len
        
        rowPtr[0] = int(rowPtr[0])
        tile_num = []
        for i in range(rowPtr_len-1):
            rowPtr[i+1] = int(rowPtr[i+1])
            tile_num.append(math.ceil((rowPtr[i+1] - rowPtr[i]) / 32))

        for i in range(colInd_len):
            colInd[i] = int(colInd[i])

        self.num_vec_row = num_vec_row
        self.col = N
        self.nnz = nnz
        self.vec_len = vec_len
        self.num_vec = num_vec
        self.tile_num = tile_num
        print("CVSE Init Done")
    def new_sparsity(self):
        # total_vec_num = self.num_vec_row * self.col
        new_sparsity = 1.0 - self.nnz/(self.num_vec * self.vec_len)
        return new_sparsity
    
    def Dense_X(self):
        Dense_X = (self.col * self.col) / (self.num_vec * self.vec_len)
        return Dense_X
    
    def all_zero_vec_cnt(self):
        total_vec_num = self.num_vec_row * self.col
        all_zero_vec_cnt = total_vec_num - self.num_vec
        return all_zero_vec_cnt
    
    def return_CSR_storage(self):
        index_type = 4#32/8=4
        value_type = 2#16/8=2
        storage = self.nnz * (value_type + index_type) + (self.num_vec_row * self.vec_len + 1) * index_type
        return storage
        
    def return_CVSE_storage(self):
        index_type = 4#32/8=4
        value_type = 2#16/8=2
        storage = self.num_vec * self.vec_len * value_type + (self.num_vec_row + 1) * index_type + self.num_vec * index_type
        return storage
    
    def return_TCF_storage(self):
        index_type = 4#32/8=4
        value_type = 2#16/8=2
        storage = self.nnz * value_type + self.nnz * 3 * index_type + (self.num_vec_row * self.vec_len + 1) * index_type
        for i in range(self.num_vec_row):
            storage += self.tile_num[i] * 32 * index_type
        return storage

    def return_SRBCSR_storage(self):
        index_type = 4#32/8=4
        value_type = 2#16/8=2
        storage = self.num_vec * self.vec_len * value_type + (self.num_vec_row + 1) * index_type
        for i in range(self.num_vec_row):
            storage += self.tile_num[i] * 32 * index_type
        return storage
        
def generate_suitesparse_csv_file(src_dir, paths_txt, vec_len):
    # paths = src_dir + paths_txt
    file_name = "./csv_data/RowMerge_suit_V" + str(vec_len) + ".csv"
    fcsv = open(file_name, 'a')
    writer = csv.writer(fcsv)
    pattern = re.compile(r'/([^/]+)\.mtx$')
    with open(paths_txt, 'r') as file:
        print("Begin to gen csv-file, vecLength=", vec_len)
        lines = [line.rstrip('\n') for line in file]
        file_num = len(lines)
        print("Number of file is ",file_num)
        for i in range(file_num):
            # filename = file.readline().split()[0]
            source_file_name = src_dir + lines[i]
            suit_name = pattern.search(lines[i]).group(1)
            source_file_name = source_file_name.replace(".mtx", ".smtx")
            print("suitsparse file name: ", suit_name)

            suite_M = suit_cvse(source_file_name, vec_len)
            NNZ = suite_M.nnz
            COL = suite_M.col
            new_sparsity = suite_M.new_sparsity()
            CSR_storage = suite_M.return_CSR_storage()
            CVSE_storage = suite_M.return_CVSE_storage()
            TCF_storage  = suite_M.return_TCF_storage()
            SR_BCSR_storage = suite_M.return_SRBCSR_storage()
            Dense_X = suite_M.Dense_X()
            this_csv_row = [suit_name, str(NNZ), str(COL), str(new_sparsity), str(CSR_storage), str(CVSE_storage), str(TCF_storage), str(SR_BCSR_storage), str(Dense_X)]
            print("New_Sparsity:\t", new_sparsity)
            # print(m1.all_zero_vec_cnt())
            print("CSR_storage:\t", CSR_storage)
            print("CVSE_storage:\t", CVSE_storage)
            print("TCF_storage:\t", TCF_storage)
            print("SR_BCSR_storage: ", SR_BCSR_storage)
            DESNE_storage = COL * COL * 2
            print("DESNE_storage:\t", DESNE_storage)
            print("Dense_X:\t", Dense_X)
            writer.writerow(this_csv_row)
            print("File-", i, " Finished")

if __name__ == "__main__":
    
    
    files = ["circuit204", "cnae9_10NN", "collins_15NN", "cop20k_A", "mosfet2", "wiki-Vote", "filter3D", "web-Google", "m133-b3", "poisson3Da", "mario002", "2cubes_sphere", "scircuit", "ca-CondMat", "p2p-Gnutella31", "psmigr_1"]
    # files = ["m133-b3", "poisson3Da", "mario002", "2cubes_sphere", "scircuit", "ca-CondMat", "p2p-Gnutella31", "psmigr_1"]
    for i in range(len(files)):
        file_name = files[i]
        path = "../benchmark/dlmc/suitesparse/" + file_name + ".mtx"
        suite_M = suite_matrix(path)
        print(suite_M.M, suite_M.K)
        print(suite_M.nnz)
        print(len(suite_M.row_indices))
        suite_M.coo2csr(1)# write csr file
        # suite_M.csr2cvse(8)
        colInd_list_array = suite_M.colInd_list_array()
        cvse8_M = sparse_matrix_format.CVSE(suite_M.M, suite_M.K, suite_M.nnz, colInd_list_array, 8)
        cvse16_M = sparse_matrix_format.CVSE(suite_M.M, suite_M.K, suite_M.nnz, colInd_list_array, 16)
        newdir8 = "../benchmark/dlmc-v8/suitesparse/" + file_name + ".smtx"
        newdir16 = "../benchmark/dlmc-v16/suitesparse/" + file_name + ".smtx"
        cvse8_M.CVSE_out(newdir8)
        cvse16_M.CVSE_out(newdir16)
        print("CVSE-file ", i, " Finished")

