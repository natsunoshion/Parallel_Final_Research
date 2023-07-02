import numpy as np
import numba as nb
import time

@nb.jit(nopython=True,parallel=True,fastmath=True)
def LU(A):
    n = A.shape[0]
    for k in range(n):
        for j in nb.prange(k + 1, n):
            A[k, j] /= A[k, k]
        A[k, k] = 1.0
        for i in nb.prange(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= A[k, j] * A[i, k]
            A[i, k] = 0

def print_matrix(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            print(A[i, j], end="\t")
        print()

sizes = [200, 500, 1000, 2000, 3000]
for n in sizes:
    # 初始化二维数组并生成随机数
    A = np.random.rand(n, n).astype(np.float32)

    # 使用Numba并行计算LU分解
    start = time.time()
    LU(A)
    end = time.time()
    diff = end - start
    print(f"Size = {n}: {diff * 1000}ms")

    # # 打印并验证计算结果
    # print("Result:")
    # print_matrix(A)
    # print()
    # break
