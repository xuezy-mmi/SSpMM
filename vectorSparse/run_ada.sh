make -f Makefile_ada clean
make -f Makefile_ada spmm_test
make -f Makefile_ada spmm_test_run_cublas
make -f Makefile_ada spmm_test_run_cusparse
make -f Makefile_ada spmm_test_run_sputnik
make -f Makefile_ada spmm_test_run_cublas_suit
make -f Makefile_ada spmm_test_run_cusparse_suit
make -f Makefile_ada spmm_test_run_sputnik_suit
make -f Makefile_ada spmm_test_run_vectorsparse_structured
make -f Makefile_ada spmm_test_run_vectorsparse_v1
make -f Makefile_ada spmm_test_run_vectorsparse_v8
cp ./data/* ../result-data/