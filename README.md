# SSpMM

## Env
```shell
g++-7.5.0
cuda-11.8
sputnik
```
## Install Sputnik
```shell
git clone https://github.com/google-research/sputnik.git
cd sputnik
cd third_party
git clone https://github.com/abseil/abseil-cpp.git
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git
cd ..
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75;80;89"
make -j12
cd ../..
```

### Get Dataset
```shell
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
rm dlmc.tar.gz
```
### Run Sparse Vector Compression (SVC) and Produce New Dataset
```shell
cd RowMerge
bash mkdir.sh
python row_merge.py
python row_merge_reorder.py
cp ./csv_data/* ../data/
cd ..
```
wait about 6hours
IF you don't want to wait, you can download the data(dlmc-v8, dlmc-v16, dataset-v8, dataset-v16) from [https://github.com/xuezy-mmi/dlmc.git](https://github.com/xuezy-mmi/dlmc.git)(TBD)
## SSpMM
```shell
cd SSpMM
```
### Build and Run
You need to change the NVCC direction (line 1) in "Makefile_xxx". (Four Makfile-files: Makefile_volta, Makefile_turing, Makefile_ampere, Makefile_ada)
We have 4 shell script files, aimed to run on different GPU architectures.
xxx depends on your GPU architexture. xxx includes volta, turing, ampere, ada.
```shell
bash ./run_xxx.sh
```
After programming executing, some csv files will be generated in ./data/. And the shell scropt has cpoied them to ../plt/ for plot figures.

## Baseline
```shell
cd vectorsparse
```
You need to edit "MakeFile_xxx", change the NVCC direction (line 1) and your Sputnik direction (line 2).
### Build and Run
```shell
bash ./run_baseline_xxx.sh
```

## Plot the Figures of Experimental Results
```shell
cd plt
bash ./plt_all.sh
```
After programming executing, all pdf files of figures will be generated in ./fig/.
