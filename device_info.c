#include <stdio.h>
#include <stdlib.h>
//#include <CL/cl.h>
#include <OpenCL/cl.h>

void print_device_info(cl_device_id device) {
    char device_name[256];
    char vendor_name[256];
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_uint compute_units;
    cl_uint max_work_group_size;
    
    // Récupérer le nom de l'appareil
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    // Récupérer le nom du fournisseur
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    // Récupérer la mémoire globale
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    // Récupérer la mémoire locale
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    // Récupérer le nombre d'unités de calcul
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    // Récupérer la taille maximale des groupes de travail
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

    // Afficher les informations récupérées
    printf("=== Informations sur le GPU ===\n");
    printf("Nom de l'appareil      : %s\n", device_name);
    printf("Fournisseur            : %s\n", vendor_name);
    printf("Mémoire globale        : %lu Mo\n", global_mem_size / (1024 * 1024));
    printf("Mémoire locale         : %lu Ko\n", local_mem_size / 1024);
    printf("Unités de calcul       : %u\n", compute_units);
    printf("Taille max. du groupe  : %u\n", max_work_group_size);
    printf("===============================\n");
}

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    // Récupérer la première plateforme
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur : impossible de récupérer la plateforme OpenCL\n");
        return -1;
    }

    // Récupérer le premier appareil (GPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Erreur : impossible de récupérer le GPU\n");
        return -1;
    }

    // Afficher les informations du GPU
    print_device_info(device);

    return 0;
}
