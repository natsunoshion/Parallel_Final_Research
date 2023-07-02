import numpy as np
from numba import njit, prange

# 定义样例参数字典
example_params = {
    3: {
        "num_columns": 562,
        "num_elimination_rows": 170,
        "num_eliminated_rows": 53
    },
    4: {
        "num_columns": 1011,
        "num_elimination_rows": 539,
        "num_eliminated_rows": 263
    },
    5: {
        "num_columns": 2362,
        "num_elimination_rows": 1226,
        "num_eliminated_rows": 453
    },
    6: {
        "num_columns": 3799,
        "num_elimination_rows": 2759,
        "num_eliminated_rows": 1953
    },
    7: {
        "num_columns": 8399,
        "num_elimination_rows": 6375,
        "num_eliminated_rows": 4535
    }
}

# 输入样例编号
example = 6

# 根据样例编号获取对应的参数值
params = example_params[example]
num_columns = params["num_columns"]
num_elimination_rows = params["num_elimination_rows"]
num_eliminated_rows = params["num_eliminated_rows"]

# 根据参数值获取数组长度
length = int(np.ceil(num_columns / 5.0))

# 使用numpy数组存储R和E
R = np.zeros((10000, length), dtype=np.uint8)
E = np.zeros((num_eliminated_rows, length), dtype=np.uint8)

# 记录是否被抬高
lifted = np.zeros(num_eliminated_rows, dtype=bool)

# 打印当前位图到屏幕
def print_bitmap():
    for i in range(num_eliminated_rows):
        for j in range(num_columns - 1, -1, -1):
            if ((E[i][j // 5] >> (j - 5 * (j // 5))) & 1) == 1:
                print(j, end=' ')
        print()

# 特殊高斯消去的 numba 并行化
@njit(parallel=True)
def solve(E, R, length, num_columns, num_eliminated_rows, lifted):
    for n in range(length - 1, -1, -1):
        records = []
        for i in range(num_eliminated_rows):
            if lifted[i]:
                continue
            for j in range(5 * n + 4, 5 * n - 1, -1):
                if (E[i][n] >> (j - 5 * n)) == 1:
                    # 有对应的消元子，那么消去
                    if R[j][n] != 0:
                        E[i][n] ^= R[j][n]
                        records.append((i, j))
                    # 否则，进行升格操作，需要立刻进行升格
                    else:
                        for pair in records:
                            row = pair[0]
                            leader = pair[1]
                            if row == i:
                                for k in range(n - 1, -1, -1):
                                    E[i][k] ^= R[leader][k]
                        R[j][:n + 1] = E[i][:n + 1]
                        lifted[i] = True
                        break

        # 根据记录进行并行化
        for m in prange(n - 1):
            for pair in records:
                row = pair[0]
                leader = pair[1]
                if lifted[row]:
                    continue
                E[row][m] ^= R[leader][m]

# 读取R元素
with open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例{} 矩阵列数{}，非零消元子{}，被消元行{}/消元子.txt".format(example, num_columns, num_elimination_rows, num_eliminated_rows), "r") as file_R:
    for line in file_R:
        bits = list(map(int, line.split()))
        index = bits[0]
        for bit in bits[1:]:
            R[index][bit // 5] |= 1 << (bit - (bit // 5) * 5)

# 读取E元素
with open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例{} 矩阵列数{}，非零消元子{}，被消元行{}/被消元行.txt".format(example, num_columns, num_elimination_rows, num_eliminated_rows), "r") as file_E:
    index = 0
    for line in file_E:
        bits = list(map(int, line.split()))
        for bit in bits:
            E[index][bit // 5] |= 1 << (bit - (bit // 5) * 5)
        index += 1

# 计时并运行高斯消去
import time
start = time.time()
solve(E, R, length, num_columns, num_eliminated_rows, lifted)
end = time.time()
diff = end - start
print(diff, "s")

# 验证结果的正确性
# print_bitmap()
