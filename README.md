# spmv_csr


clang -framework OpenCL -o spmv_coo spmv_coo.c

clang -framework OpenCL  -o device_info device_info.c

gcc -o csr_matvec csr_matvec.c -framework OpenCL
