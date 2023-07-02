#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <mpi.h>
#include <chrono>

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// 线程数
#define NUM_THREADS 4

// 数组长度
const int length = ceil(num_columns / 5.0);

// R：消元子，E：被消元行
char R[10000][length];
char E[num_eliminated_rows][length];

// 记录 <消元行操作的行号，首项所在的列号>
struct Pair {
    int first;
    int second;
};

// 记录是否升格
bool lifted[num_eliminated_rows];

// 将当前位图打印到屏幕上
void print() {
    for (int i = 0; i < num_eliminated_rows; i++) {
        // cout << i << ':';
        for (int j = num_columns - 1; j >= 0; j--) {
            // 第j位为1
            if ((((E[i][j / 5] >> (j - 5 * (j / 5)))) & 1) == 1) {
                cout << j << ' ';
            }
        }
        cout << endl;
    }
}

// 特殊高斯消去法并行实现，假设每一轮取5列消元子/被消元行出来
void solve(int myid, int num) {
    // 定义 MPI 的 Pair 结构体类型
    MPI_Datatype mpi_pair_type;
    MPI_Type_contiguous(2, MPI_INT, &mpi_pair_type);
    MPI_Type_commit(&mpi_pair_type);
    int n;
    // MPI 多进程
    for (n = length - 1; n >= 0; n--) {
        // 使用结构体记录，方便发送数据
        Pair records[100000];
        int recordsCount = 0;
        if (n % num == myid) {
            // 遍历被消元行
            for (int i = 0; i < num_eliminated_rows; i++) {
                // 不处理升格的那些行
                if (lifted[i]) {
                    continue;
                }
                // 找首项消去，必须从高到低找
                for (int j = 5 * (n + 1) - 1; j >= 5 * n; j--) {
                    if (E[i][n] >> (j - 5 * n) == 1) {
                        if (R[j][n] != 0) {
                            E[i][n] ^= R[j][n];
                            records[recordsCount] = {i, j};
                            recordsCount++;
                        } else {
                            // 立刻升格，方便之后其他行消去
                            for (auto pair : records) {
                                int row = pair.first;
                                int leader = pair.second;
                                if (row == i) {
                                    // 补上剩下位的异或
                                    for (int k = n - 1; k >= 0; k--) {
                                        E[i][k] ^= R[leader][k];
                                    }
                                }
                            }
                            // 消元子第j行 = 被消元行第i行
                            for (int k = 0; k < length; k++) {
                                R[j][k] = E[i][k];
                            }
                            lifted[i] = true;
                            break;
                        }
                    }
                }
            }

            // 发送消息
            for (int j = 0; j < num; j++) {
                if (j != myid) {
                    // 发送记录数和记录
                    MPI_Send(&recordsCount, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    MPI_Send(records, recordsCount, mpi_pair_type, j, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            // 其余进程接收消息
            MPI_Recv(&recordsCount, 1, MPI_INT, n % num, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(records, recordsCount, mpi_pair_type, n % num, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 接下来，对剩下的列进行并行计算
        for (int m = n - 1; m >= 0; m--) {
            if (m % num == myid) {
                for (int i = 0; i < recordsCount; i++) {
                    int row = records[i].first;
                    int leader = records[i].second;
                    // 跳过已经升格的行
                    if (lifted[row]) {
                        continue;
                    }
                    E[row][m] ^= R[leader][m];

                    // 发送消息
                    for (int j = 0; j < num; j++) {
                        if (j != myid) {
                            MPI_Send(&E[row][m], 1, MPI_CHAR, j, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            } else {
                // 其余进程接收消息
                for (int i = 0; i < recordsCount; i++) {
                    int row = records[i].first;
                    int leader = records[i].second;
                    // 跳过已经升格的行
                    if (lifted[row]) {
                        continue;
                    }
                    MPI_Recv(&E[row][m], 1, MPI_CHAR, m % num, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int myid, numprocs;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("/home/data/Groebner/4_1011_539_263/1.txt");
    // file_R.open("R.txt");
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

    // 读入被消元行
    ifstream file_E;
    file_E.open("/home/data/Groebner/4_1011_539_263/2.txt");
    // file_E.open("E.txt");

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

    // 使用C++11的chrono库来计时
    auto start = chrono::high_resolution_clock::now();
    solve(myid, numprocs);
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);

    if (myid == 0) {
        cout << diff.count() << "ms" << endl;
        // 验证结果正确性
        // print();
    }

    MPI_Finalize();
    return 0;
}
