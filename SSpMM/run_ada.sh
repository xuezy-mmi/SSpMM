make -f Makefile_ada clean
make -f Makefile_ada sspmm_test
make -f Makefile_ada tspmm_test
make -f Makefile_ada sspmm_test_run800
make -f Makefile_ada sspmm_test_run801
make -f Makefile_ada sspmm_test_run1600
make -f Makefile_ada sspmm_test_run1601
make -f Makefile_ada sspmm_test_run810
make -f Makefile_ada sspmm_test_run811
make -f Makefile_ada sspmm_test_run1610
make -f Makefile_ada sspmm_test_run1611
make -f Makefile_ada tspmm_test_run
# make -f Makefile_ada sspmm_test_runsuit8
# make -f Makefile_ada sspmm_test_runsuit16
cp ./data/* ../result-data/