export CUDA_VISIBLE_DEVICES=4
./out/sspmm_test ./paths.txt 512  8 2 0 0 1 1 0 ampere
./out/sspmm_test ./paths.txt 64   8 2 0 0 1 1 0 ampere
./out/sspmm_test ./paths.txt 4096 8 2 0 0 1 1 0 ampere
./out/sspmm_test ./paths.txt 512 16 2 0 0 1 1 0 ampere