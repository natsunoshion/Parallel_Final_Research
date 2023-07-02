/*
使用char数组实现位图
5个bit组成一个int
基本就只能分析代码调试，非常阴间
*/

#include <bits/stdc++.h>

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// 线程数
#define NUM_THREADS 4

// 数组长度
const int length = ceil(num_columns / 5.0);

// 使用char数组进行存储，R：消元子，E：被消元行
char R[10000][length];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                        // 所以不能直接用num_elimination_rows设置数组大小
char E[num_eliminated_rows][length];

// 记录 <消元行操作的行号，首项所在的列号>
vector<pair<int, int>> records;

// 记录是否升格
bool lifted[num_eliminated_rows];

// 将当前位图打印到屏幕上
void print() {
    for (int i=0; i<num_eliminated_rows; i++) {
        // cout << i << ':';
        for (int j=num_columns-1; j>=0; j--) {
            // 第j位为1
            if (((E[i][j / 5] >> (j - 5*(j/5)))) & 1 == 1) {
                cout << j << ' ';
            }
        }
        // for (int j=length-1; j>=0; j--) {
        //     cout << E[i][j] << ' ';
        // }
        cout << endl;
    }
}

// 特殊高斯消去法并行OpenMP实现，假设每一轮取5列消元子/被消元行出来
void solve() {
    // cout << "in";
    int n;
    // 每一次遍历消元子、被消元行的5个bit，通过数组的一个元素来实现
    // E[i][x]对应了5x ~ 5(x+1)-1这5个bit，从右向左存储bit
    // 这样存储的好处：不用考虑边界
    // OpenMP多线程并行化，循环外创建
    #pragma omp parallel num_threads(NUM_THREADS), shared(records), private(n)  // 这个private(n)很重要
    for (n = length - 1; n >= 0; n--) {
        #pragma omp single
        {
            records.clear();
            // 遍历被消元行
            for (int i=0; i<num_eliminated_rows; i++) {
                // 不处理升格的那些行
                if (lifted[i]) {
                    continue;
                }
                // 找首项消去，必须从高到低找
                for (int j = 5*(n+1)-1; j >= 5*n; j--) {
                    if (E[i][n] >> (j-5*n) == 1) {
                        if (R[j][n] != 0) {
                            E[i][n] ^= R[j][n];
                            records.emplace_back(i, j);
                        } else {
                            // 立刻升格，方便之后其他行消去
                            for (auto pair : records) {
                                int row = pair.first;
                                int leader = pair.second;
                                if (row == i) {
                                    // 补上剩下位的异或
                                    for (int k=n-1; k>=0; k--) {
                                        E[i][k] ^= R[leader][k];
                                    }
                                }
                            }
                            // 消元子第j行 = 被消元行第i行
                            for (int k=0; k<length; k++) {
                                R[j][k] = E[i][k];
                            }
                            lifted[i] = true;
                            break;
                        }
                    }
                }
            }
        }
        // 接下来，对剩下的列进行并行计算，按照records中的记录进行多线程操作
        // 这些数组元素就按照记录挨个异或，由于已经有记录了（无依赖），所以可以并行化
        #pragma omp for simd schedule(guided, 1)
        for (int m=n-1; m>=0; m--) {
            for (auto pair : records) {
                int row = pair.first;
                int leader = pair.second;
                // 跳过已经升格的行
                if (lifted[row]) {
                    continue;
                }
                E[row][m] ^= R[leader][m];
            }
        }
        // cout << "调试开始" << endl;
        // cout << "n=" << n << endl;
        // print();
    }
}

int main() {
    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("R.txt");
    if (file_R.fail()) {
        cout << "读入失败" << endl;
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
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
//--------------------------------
    // 读入被消元行
    ifstream file_E;
    // file_E.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/被消元行.txt");
    file_E.open("E.txt");

    // 被消元行的index就是读入的行数
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        while (line >> bit) {
            // 将第index行的被消元行对应位 置1
            E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
        }
        index++;
    }
    // cout << E[6][50];
    // print();
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