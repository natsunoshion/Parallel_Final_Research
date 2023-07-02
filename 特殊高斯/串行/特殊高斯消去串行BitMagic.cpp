#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include "bm/bm.h"

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// 使用BitMagic进行存储，R：消元子，E：被消元行
bm::bvector<> R[10000];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                               // 所以不能直接用num_elimination_rows设置数组大小
bm::bvector<> E[num_eliminated_rows];

// 特殊高斯消去法串行实现
void solve() {
    // 循环处理每个被消元行
    for (int i = 0; i < num_eliminated_rows; i++) {
        // 如果当前被消元行为零，则直接跳过
        if (E[i].none()) {
            continue;
        }

        // 循环处理当前被消元行的每一项
        while (!E[i].none()) {
            // 找到当前被消元行的首项
            int k = num_columns - 1;
            while (E[i][k]==0 && k>=0) {
                k--;
            }
            // cout << "首项" << k;

            // 如果首项对应的消元子不存在，则进行“升格”，将当前行加入该消元子的集合中
            if (!R[k].any()) {
                R[k] = E[i];
                // E[i].reset();
                // cout << "升格";
                break;
            }
            // 如果首项对应的消元子存在，则进行消去操作
            else {
                E[i] ^= R[k];
            }
        }
    }
}

void print() {
    for (int i=0; i<num_eliminated_rows; i++) {
        // cout << i << ':';
        for (int j=num_columns-1; j>=0; j--) {
            if (E[i][j] == 1) {
                cout << j << ' ';
            }
        }
        cout << endl;
    }
}

int main() {
    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/消元子.txt");
    if (file_R.fail()) {
        cout << "读入失败" << endl;
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        int first_in = 1;

        // 消元子的索引是其首项
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }

            // 将第index行的消元子bitset对应位 置1
            R[index][bit] = 1;
        }
    }
    file_R.close();
//--------------------------------
    // 读入被消元行
    ifstream file_E;
    // file_E.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/被消元行.txt");
    file_E.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/被消元行.txt");

    // 被消元行的索引就是读入的行数
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        while (line >> bit) {
            // 将第index行的消元子bitset对应位 置1
            E[index][bit] = 1;
        }
        index++;
    }
//--------------------------------
    // 使用C++11的chrono库来计时
    auto start = chrono::high_resolution_clock::now();
    solve();
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << diff.count() << "ms" << endl;
//--------------------------------
    // 验证结果正确性
    // print();
    return 0;
}