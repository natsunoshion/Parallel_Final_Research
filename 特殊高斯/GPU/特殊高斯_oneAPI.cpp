%%writefile lab/matrix.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

namespace s = sycl;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 254
#define num_elimination_rows 106
#define num_eliminated_rows 53

// 数组长度
const int length = 51;

// 使用char数组进行存储，R：消元子，E：被消元行
char R[10000][length];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                        // 所以不能直接用num_elimination_rows设置数组大小
char E[num_eliminated_rows][length];

// 记录是否升格
bool lifted[num_eliminated_rows];

// 将当前位图打印到屏幕上
void print() {
    for (int i=0; i<num_eliminated_rows; i++) {
        // std::cout << i << ':';
        for (int j=num_columns-1; j>=0; j--) {
            // 第j位为1
            if ((((E[i][j / 5] >> (j - 5*(j/5)))) & 1) == 1) {
                std::cout << j << ' ';
            }
        }
        // for (int j=length-1; j>=0; j--) {
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
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/R.txt");
    if (file_R.fail()) {
        std::cout << "读入失败" << std::endl;
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
    // file_E.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/被消元行.txt");
    file_E.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/E.txt");

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

// 特殊高斯消去法并行 char 数组实现
void solve() {
    // 创建队列和设备选择器
    s::queue q{s::default_selector{}};

    // 创建缓冲区
    s::buffer<char, 2> E_buf(s::range<2>{num_eliminated_rows, length});
    s::buffer<char, 2> R_buf(s::range<2>{10000, length});

    // 将数据从主机内存传输到设备缓冲区
    {
        auto E_acc = E_buf.get_access<s::access::mode::discard_write>();
        auto R_acc = R_buf.get_access<s::access::mode::discard_write>();

        for (int i = 0; i < num_eliminated_rows; i++) {
            for (int j = 0; j < length; j++) {
                E_acc[i][j] = E[i][j];
            }
        }

        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < length; j++) {
                R_acc[i][j] = R[i][j];
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 提交任务并执行内核
    q.submit([&](s::handler& h) {
        auto E_acc = E_buf.get_access<s::access::mode::read_write>(h);
        auto R_acc = R_buf.get_access<s::access::mode::read_write>(h);

        h.parallel_for(s::range<1>{num_eliminated_rows}, [=](s::id<1> idx) {
            int i = idx[0];
            bool is_eliminated = false;

            for (int j = length - 1; j >= 0; j--) {
                for (int k = 4; k >= 0; k--) {
                    if ((E_acc[i][j] >> k) == 1) {
                        int leader = 5 * j + k;
                        if (R_acc[leader][j] != 0) {
                            for (int m = j; m >= 0; m--) {
                                E_acc[i][m] ^= R_acc[leader][m];
                            }
                        } else {
                            for (int m = j; m >= 0; m--) {
                                R_acc[leader][m] = E_acc[i][m];
                            }
                            is_eliminated = true;
                            break; // 跳出内层循环
                        }
                    }
                }
                if (is_eliminated) {
                    break; // 跳出外层循环
                }
            }
        });
    });

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << diff.count() << "ms" << std::endl;

    // 将数据从设备缓冲区传输回主机内存
    {
        auto E_acc = E_buf.get_access<s::access::mode::read>();
        auto R_acc = R_buf.get_access<s::access::mode::read>();

        for (int i = 0; i < num_eliminated_rows; i++) {
            for (int j = 0; j < length; j++) {
                E[i][j] = E_acc[i][j];
            }
        }

        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < length; j++) {
                R[i][j] = R_acc[i][j];
            }
        }
    }
}

int main() {
    input();
    solve();

    return 0;
}
