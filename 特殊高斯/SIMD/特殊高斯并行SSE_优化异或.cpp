#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <nmmintrin.h>
#include <immintrin.h>

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// 数组长度
const int length = 203;

// 使用char数组进行存储，R：消元子，E：被消元行
int R[10000][length];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                        // 所以不能直接用num_elimination_rows设置数组大小
int E[num_eliminated_rows][length];

// 记录是否升格
bool lifted[num_eliminated_rows];

// 将当前位图打印到屏幕上
void print() {
    for (int i = 0; i < num_eliminated_rows; i++) {
        // std::cout << i << ':';
        for (int j = num_columns - 1; j >= 0; j--) {
            // 第j位为1
            if ((((E[i][j / 5] >> (j - 5 * (j / 5)))) & 1) == 1) {
                std::cout << j << ' ';
            }
        }
        // for (int j = length - 1; j >= 0; j--) {
        //     std::cout << E[i][j] << ' ';
        // }
        std::cout << std::endl;
    }
}

// 读入
void input() {
    // 读入消元子
    std::ifstream file_R;
    char buffer[10000] = {0};
    file_R.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/消元子.txt");
    // file_R.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/R.txt");
    if (file_R.fail()) {
        std::cout << "Failed to open file" << std::endl;
        perror("File error");
        std::string command = "dir \"D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/消元子.txt\"";
        int result = std::system(command.c_str());
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        std::stringstream line(buffer);
        int first_in = 1;

        // 消元子的index是其首项
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }

            // 将第index行的消元子对应位 置1
            R[index][bit / 5] |= 1 << (bit - (bit / 5) * 5);
        }
    }
    file_R.close();

    // 读入被消元行
    std::ifstream file_E;
    file_E.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/被消元行.txt");
    // file_E.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/E.txt");

    // 被消元行的index就是读入的行数
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        std::stringstream line(buffer);
        while (line >> bit) {
            // 将第index行的被消元行对应位 置1
            E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
        }
        index++;
    }
}

// 特殊高斯消去法串行 int 数组实现
void solve() {
    auto start = std::chrono::high_resolution_clock::now();

    // 遍历被消元行：逐元素消去，一次清空
    for (int i = 0; i < num_eliminated_rows; i++) {
        bool is_eliminated = false;
        // 行内遍历依次找首项消去
        for (int j = length - 1; j >= 0; j--) {
            // cout << j << "消去" << endl;
            for (int k = 4; k >= 0; k--) {
                if ((E[i][j] >> k) == 1) {
                    // 获得首项
                    int leader = 5 * j + k;
                    if (R[leader][j] != 0) {
                        // 使用 SIMD 优化
                        int m;
                        for (m = j; m - 4 >= 0; m -= 4) {
                            __m128i e = _mm_loadu_si128((__m128i*)&E[i][m]);
                            __m128i r = _mm_loadu_si128((__m128i*)&R[leader][m]);

                            // 对整行进行异或操作
                            e = _mm_xor_si128(e, r);

                            _mm_storeu_si128((__m128i*)&E[i][m], e);
                        }
                        for (; m >= 0; m--) {
                            E[i][m] ^= R[leader][m];
                        }
                    } else {
                        // 否则升格，升格之后这一整行都可以不用管了
                        for (int m = j; m >= 0; m--) {
                            R[leader][m] = E[i][m];
                        }
                        // 跳出多重循环
                        is_eliminated = true;
                    }
                }
                if (is_eliminated) {
                    break;
                }
            }
            if (is_eliminated) {
                break;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << diff.count() << "ms" << std::endl;
}

int main() {
    input();
    solve();
    // print();
    return 0;
}
