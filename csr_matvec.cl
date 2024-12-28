__kernel void csr_matvec(
    __global const int* row_ptr,
    __global const int* col_idx,
    __global const float* values,
    __global const float* vector,
    __global float* result,
    const int num_rows
) {
    int row = get_global_id(0);
    
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * vector[col_idx[j]];
        }
        
        result[row] = sum;
    }
}
