import numpy as np
import time

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
example = 3

# 根据样例编号获取对应的参数值
params = example_params[example]
num_columns = params["num_columns"]
num_elimination_rows = params["num_elimination_rows"]
num_eliminated_rows = params["num_eliminated_rows"]

# 根据参数值获取数组长度
length = int(np.ceil(num_columns / 5.0))

# 使用numpy数组进行存储，R：消元子，E：被消元行
R = np.zeros((10000, length), dtype=np.uint32)
E = np.zeros((num_eliminated_rows, length), dtype=np.uint32)

# 记录是否升格
lifted = np.zeros(num_eliminated_rows, dtype=bool)

# 将当前位图打印到屏幕上
def print_bitmap():
    for i in range(num_eliminated_rows):
        print(f"{i}: ", end="")
        for j in range(num_columns - 1, -1, -1):
            if ((E[i][j // 5] >> (j - 5 * (j // 5))) & 1) == 1:
                print(j, end=" ")
        print()

# 读入
def input_data():
    # 读入消元子
    file_R = open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例{} 矩阵列数{}，非零消元子{}，被消元行{}/消元子.txt".format(example, num_columns, num_elimination_rows, num_eliminated_rows), "r")
    for line in file_R:
        bits = list(map(int, line.split()))
        first_in = True
        index = 0
        for bit in bits:
            if first_in:
                first_in = False
                index = bit
            R[index][bit // 5] |= 1 << (bit - (bit // 5) * 5)
    file_R.close()

    # 读入被消元行
    file_E = open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例{} 矩阵列数{}，非零消元子{}，被消元行{}/被消元行.txt".format(example, num_columns, num_elimination_rows, num_eliminated_rows), "r")
    index = 0
    for line in file_E:
        bits = list(map(int, line.split()))
        for bit in bits:
            E[index][bit // 5] |= 1 << (bit - (bit // 5) * 5)
        index += 1
    file_E.close()

# 特殊高斯消去法串行 numpy 数组实现
def solve():
    start = time.time()

    # 遍历被消元行：逐元素消去，一次清空
    for i in range(num_eliminated_rows):
        is_eliminated = False
        # 行内遍历依次找首项消去
        for j in range(length - 1, -1, -1):
            for k in range(4, -1, -1):
                if (E[i][j] >> k) == 1:
                    # 获得首项
                    leader = 5 * j + k
                    if R[leader][j] != 0:
                        # 有消元子就消去，需要对整行异或
                        for m in range(j, -1, -1):
                            E[i][m] ^= R[leader][m]
                    else:
                        # 否则升格，升格之后这一整行都可以不用管了
                        R[leader][:j + 1] = E[i][:j + 1]
                        is_eliminated = True
                        break
            if is_eliminated:
                break

    end = time.time()
    print(f"{end - start}s")

if __name__ == '__main__':
    input_data()
    # print_bitmap()
    solve()
    # print_bitmap()
