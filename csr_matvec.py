import pyopencl as cl
import numpy as np

# --- Données CSR
row_ptr = np.array([0, 3, 4, 7, 9], dtype=np.int32)
col_idx = np.array([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np.int32)
values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y = np.zeros(4, dtype=np.float32)

# --- Initialisation OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# --- Kernel OpenCL pour produit CSR × vecteur
kernel_code = """
__kernel void spmv_csr(
    __global const int *row_ptr,
    __global const int *col_idx,
    __global const float *values,
    __global const float *x,
    __global float *y)
{
    int row = get_global_id(0);
    float dot = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int i = row_start; i < row_end; i++) {
        dot += values[i] * x[col_idx[i]];
    }
    y[row] = dot;
}
"""

# --- Compilation et buffers
program = cl.Program(ctx, kernel_code).build()

mf = cl.mem_flags
row_ptr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_ptr)
col_idx_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=col_idx)
values_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes)

# --- Exécution
program.spmv_csr(
    queue, (y.size,), None,
    row_ptr_buf, col_idx_buf, values_buf, x_buf, y_buf
)

# --- Récupération résultat
cl.enqueue_copy(queue, y, y_buf)
queue.finish()

# --- Affichage
print("Résultat y =", y)

