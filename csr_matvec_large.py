import numpy as np
import pyopencl as cl

# Paramètres : grande matrice
m, n = 100_000, 100_000
density = 0.0001  # faible densité (~0.01%)
nnz = int(m * n * density)


print(f"m,n= ",m,",",n)
print("nnz=",nnz)

# Génère une matrice CSR aléatoire
rng = np.random.default_rng(42)

row_ptr = np.zeros(m + 1, dtype=np.int32)
nnz_per_row = np.random.poisson(lam=nnz / m, size=m)
nnz_per_row = np.clip(nnz_per_row, 0, n)  # pas plus que n

nnz = np.sum(nnz_per_row)
row_ptr[1:] = np.cumsum(nnz_per_row)

col_idx = rng.integers(low=0, high=n, size=nnz, dtype=np.int32)
values = rng.random(size=nnz, dtype=np.float32) * 10.0
x = rng.random(size=n, dtype=np.float32)
y = np.zeros(m, dtype=np.float32)

# Choix plateforme GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# OpenCL Kernel
kernel_code = """
__kernel void spmv_csr(
    __global const int *row_ptr,
    __global const int *col_idx,
    __global const float *values,
    __global const float *x,
    __global float *y)
{
    int row = get_global_id(0);
    float sum = 0.0;
    for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; jj++) {
      sum += values[jj] * x[col_idx[jj]];
    }
    y[row] = sum;
}
"""

prg = cl.Program(ctx, kernel_code).build()

# Copie vers le device
mf = cl.mem_flags
d_row_ptr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_ptr)
d_col_idx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=col_idx)
d_values = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values)
d_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
d_y = cl.Buffer(ctx, mf.WRITE_ONLY, y.nbytes)

# Lancement du kernel
global_size = (m,)
prg.spmv_csr(queue, global_size, None, d_row_ptr, d_col_idx, d_values, d_x, d_y)
cl.enqueue_copy(queue, y, d_y)

# Affiche les premiers résultats
print("y[:10] =", y[:10])

