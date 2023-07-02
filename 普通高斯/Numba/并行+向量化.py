import numpy as np
import numba as nb
import time

@nb.vectorize(nopython=True, fastmath=True)
def divide_element(a, b):
    return a / b

@nb.vectorize(nopython=True, fastmath=True)
def subtract_element(a, b, c):
    return a - b * c

@nb.jit(nopython=True, parallel=True)
def LU(A):
    n = A.shape[0]
    for k in range(n):
        A[k, k] = 1.0
        A[k, k+1:n] = divide_element(A[k, k+1:n], A[k, k])
        for i in nb.prange(k + 1, n):
            A[i, k+1:n] = subtract_element(A[i, k+1:n], A[k, k+1:n], A[i, k])
            A[i, k] = 0.0

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
