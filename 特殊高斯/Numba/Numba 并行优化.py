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
example = 5

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

@njit(parallel=True)
def solve(E, R, length, num_columns, num_eliminated_rows):
    # for i in prange(num_columns-1, -1, -8):
    #     for j in range(num_eliminated_rows):
    #         for num in range(length):
    #             for index in range(j):
    #                 if E[j, length] <= i and E[j, length] >= i - 7:
    #                     if R[index, length] == 1:
    #                         for k in range(length):
    #                             E[j, k] = E[j, k] ^ R[i, k]

    #                         S_num = 0
    #                         for num in range(length):
    #                             if E[j, num] != 0:
    #                                 temp = E[j, num]
    #                                 while temp != 0:
    #                                     temp = temp >> 1
    #                                     S_num += 1
    #                                 S_num += num * 32
    #                                 break
    #                         E[j, length] = S_num - 1
    #                     else:
    #                         for k in range(length):
    #                             R[index, k] = E[j, k]
    #                         R[index, length] = 1
    #                         break

    # 不升格地处理被消元行，每轮处理 8 个消元子，范围：首项在 i-7 到 i
    for i in range(length - 1, -2, -8):
        sign = True
        while sign:
            # 遍历被消元行的每一行
            for j in range(num_eliminated_rows):
                index = R[j][num_columns]
                while i - 7 <= R[j][num_columns] <= i:
                    if E[index][num_columns] == 1:
                        # 消去
                        for k in range(num_columns):
                            R[j][k] ^= E[index][k]

                        num = 0
                        S_num = 0
                        for num in range(num_columns):
                            if R[j][num] != 0:
                                temp = R[j][num]
                                while temp != 0:
                                    temp = temp >> 1
                                    S_num += 1
                                S_num += num * 32
                                break
                        R[j][num_columns] = S_num - 1
                    else:
                        break

        # for i in range(length % 8 - 1, -1, -1):
        #     for j in range(num_eliminated_rows):
        #         while R[j][num_columns] == i:
        #             if E[i][num_columns] == 1:
        #                 for k in range(num_columns):
        #                     R[j][k] ^= E[i][k]

        #                 num = 0
        #                 S_num = 0
        #                 for num in range(num_columns):
        #                     if R[j][num] != 0:
        #                         temp = R[j][num]
        #                         while temp != 0:
        #                             temp = temp >> 1
        #                             S_num += 1
        #                         S_num += num * 32
        #                         break
        #                 R[j][num_columns] = S_num - 1
        #             else:
        #                 break

                # 然后重新判断是否结束，如果未结束则升格相应的消元子
                sign = False
                for i in range(num_eliminated_rows):
                    temp = R[i][num_columns]
                    if temp == -1:
                        continue

                    if E[temp][num_columns] == 0:
                        for k in range(num_columns):
                            E[temp][k] = R[i][k]
                        R[i][num_columns] = -1
                        sign = True

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
solve(E, R, length, num_columns, num_eliminated_rows)
end = time.time()
diff = (end - start) * 1000
print(diff, "ms")

# 验证结果的正确性
print_bitmap()
