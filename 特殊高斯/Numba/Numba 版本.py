import numpy as np
import time

# 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
num_columns = 1011
num_elimination_rows = 539
num_eliminated_rows = 263

# 数组长度
length = 203

# 使用numpy数组进行存储，R：消元子，E：被消元行
R = np.zeros((10000, length), dtype=np.uint32)  # R[i]记录了首项为i（下标从0开始记录）的消元子行
E = np.zeros((num_eliminated_rows, length), dtype=np.uint32)

# 记录是否升格
lifted = np.zeros(num_eliminated_rows, dtype=bool)

# 将当前位图打印到屏幕上
def print_bitmap():
    for i in range(num_eliminated_rows):
        for j in range(num_columns - 1, -1, -1):
            # 第j位为1
            if (((E[i][j // 5] >> (j % 5))) & 1) == 1:
                print(j, end=' ')
        print()

# 读入
def input_data():
    # 读入消元子
    file_R = open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/消元子.txt", "r")
    # file_R = open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/R.txt", "r")
    for line in file_R:
        line = line.strip()
        bits = line.split(' ')
        first_in = True

        # 消元子的index是其首项
        index = int(bits[0])
        for i in range(1, len(bits)):
            bit = int(bits[i])
            if first_in:
                first_in = False
                index = bit

            # 将第index行的消元子对应位 置1
            j = bit // 32
            k = bit % 32
            R[index][j] |= (1 << k)
    file_R.close()

    # 读入被消元行
    file_E = open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/被消元行.txt", "r")
    # file_E = open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/E.txt", "r")

    index = 0
    for line in file_E:
        line = line.strip()
        bits = line.split(' ')
        for bit in bits:
            # 将第index行的被消元行对应位 置1
            j = int(bit) // 32
            k = int(bit) % 32
            E[index][j] |= (1 << k)
        index += 1
    file_E.close()

# 特殊高斯消去法串行 numpy 数组实现
def solve():
    start_time = time.time()

    # 遍历被消元行：逐元素消去，一次清空
    for i in range(num_eliminated_rows):
        is_eliminated = False
        # 行内遍历依次找首项消去
        for j in range(length - 1, -1, -1):
            for k in range(4, -1, -1):
                if (E[i][j] >> k) & 1 == 1:
                    # 获得首项
                    leader = 5 * j + k
                    if R[leader][j] != 0:
                        # 有消元子就消去，需要对整行异或
                        for m in range(j, -1, -1):
                            E[i][m] ^= R[leader][m]
                    else:
                        # 否则升格，升格之后这一整行都可以不用管了
                        for m in range(j, -1, -1):
                            R[leader][m] = E[i][m]
                        # 跳出多重循环
                        is_eliminated = True
                if is_eliminated:
                    break
            if is_eliminated:
                break

    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"运行时间：{execution_time:.4f}ms")

input_data()
print_bitmap()
# solve()
# print_bitmap()
