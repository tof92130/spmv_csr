#include <stdio.h>
#include <stdlib.h>

//#include <CL/cl.h>
#include <OpenCL/cl.h>


// Fonction pour lire un fichier (kernel source)
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erreur : Impossible d'ouvrir le fichier %s\n", filename);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);

    char* source = (char*)malloc(size + 1);
    fread(source, sizeof(char), size, file);
    source[size] = '\0';
    fclose(file);

    return source;
}

int main() {
    // Matrice en format CSR
    const int num_rows = 4;
    const int num_cols = 4;
    const int nnz = 9;
    
    /* Exemple nvidia
      [1 0 2 3] [1]   [19]
      [0 4 0 0] [2] = [ 8]
      [5 0 6 7] [3]   [51]
      [0 8 0 9] [4]   [52]
    */
    
    int  row_ptr[] = {0, 3, 4, 7, 9};
    int  col_idx[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    float values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    
    //int  row_ptr[] = {0, 2, 4, 7, 9};
    //int  col_idx[] = {0, 1, 0, 2, 1, 2, 3, 2, 3};
    //float values[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    float vector[] = {1.0, 2.0, 3.0, 4.0};
    float result[num_rows];
    
    printf("\nAffichage intialisation vector:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", vector[i]);
    }
    
    // Initialisation OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // Obtenez la plateforme OpenCL et le GPU (M1)
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur : Impossible d'obtenir la plateforme.\n");
        return -1;
    }
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur : Impossible d'obtenir le GPU.\n");
        return -1;
    }
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue   = clCreateCommandQueue(context, device, 0, &err);

    // Lire et compiler le kernel
    char* source = read_file("csr_matvec.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    free(source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Erreur de compilation du kernel :\n%s\n", log);
        free(log);
        return -1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "csr_matvec", &err);
    
    // Création des buffers
    cl_mem row_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (num_rows+1) * sizeof(int)  , row_ptr, &err);
    cl_mem col_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nnz          * sizeof(int)  , col_idx, &err);
    cl_mem val_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nnz          * sizeof(float), values , &err);
    cl_mem vec_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_cols     * sizeof(float), vector , &err);
    cl_mem res_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY                      , num_rows     * sizeof(float), NULL   , &err);
    
    // Définitions des arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &row_buf );
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &col_buf );
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &val_buf );
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &vec_buf );
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &res_buf );
    clSetKernelArg(kernel, 5, sizeof(int)   , &num_rows);
    
    // Exécution du kernel
    size_t global_size = num_rows;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

    // Lecture des résultats
    clEnqueueReadBuffer(queue, res_buf, CL_TRUE, 0, num_rows * sizeof(float), result, 0, NULL, NULL);

    // Affichage des résultats
    printf("\nRésultat :\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", result[i]);
    }
    
    // Libération des ressources
    clReleaseMemObject(row_buf);
    clReleaseMemObject(col_buf);
    clReleaseMemObject(val_buf);
    clReleaseMemObject(vec_buf);
    clReleaseMemObject(res_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
