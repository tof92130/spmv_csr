# spmv_csr


clang -framework OpenCL  -o device_info device_info.c


gcc -o device_info device_info.c -framework OpenCL
gcc -o csr_matvec  csr_matvec.c  -framework OpenCL
