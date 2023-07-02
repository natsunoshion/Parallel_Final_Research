#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 3799
#define num_elimination_rows 2759
#define num_eliminated_rows 1953

// 数组长度
const int length = ceil(num_columns / 5.0);

// 记录是否升格
bool lifted[num_eliminated_rows];

// 特殊高斯消去法并行化
void solve_parallel(int rank, int size, char (*shared_R)[length], char (*shared_E)[length]) {
    // 计算每个进程的任务范围
    int num_eliminated_rows_per_process = num_eliminated_rows / size;
    int start_row = rank * num_eliminated_rows_per_process;
    int end_row = start_row + num_eliminated_rows_per_process;

    // 遍历被消元行：逐元素消去，一次清空
    for (int i = start_row; i < end_row; i++) {
        bool is_eliminated = false;
        // 行内遍历依次找首项消去
        for (int j = length - 1; j >= 0; j--) {
            for (int k = 4; k >= 0; k--) {
                if ((shared_E[i][j] >> k) == 1) {
                    // 获得首项
                    int leader = 5 * j + k;
                    if (shared_R[leader][j] != 0) {
                        // 有消元子就消去，需要对整行异或
                        for (int m = j; m >= 0; m--) {
                            shared_E[i][m] ^= shared_R[leader][m];
                        }
                    } else {
                        // 否则升格，升格之后这一整行都可以不用管了
                        for (int m = j; m >= 0; m--) {
                            shared_R[leader][m] = shared_E[i][m];
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
        // 若未升格，则该行已被消元，记为已消元行
        if (!is_eliminated) {
            lifted[i] = false;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char (*shared_R)[length];
    char (*shared_E)[length];

    MPI_Win win_R, win_E;

    // 在每个进程上创建共享内存
    MPI_Win_allocate(sizeof(char) * num_columns * length, sizeof(char),
                     MPI_INFO_NULL, MPI_COMM_WORLD, &shared_R, &win_R);
    MPI_Win_allocate(sizeof(char) * num_eliminated_rows * length, sizeof(char),
                     MPI_INFO_NULL, MPI_COMM_WORLD, &shared_E, &win_E);

    // 将共享内存块清零
    memset(shared_R, 0, sizeof(char) * num_columns * length);
    memset(shared_E, 0, sizeof(char) * num_eliminated_rows * length);

    // 进程0负责将R和E的初始值发送到共享内存
    if (rank == 0) {
        // 读入消元子
        ifstream file_R;
        char buffer[10000] = {0};
        file_R.open("/home/data/Groebner/6_3799_2759_1953/1.txt");
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
                shared_R[index][bit / 5] |= 1 << (bit - (bit / 5) * 5);
            }
        }
        file_R.close();

        // 读入被消元行
        ifstream file_E;
        file_E.open("/home/data/Groebner/6_3799_2759_1953/2.txt");

        // 被消元行的index就是读入的行数
        int index = 0;
        while (file_E.getline(buffer, sizeof(buffer))) {
            // 每一次读入一行，消元子每32位记录进一个int中
            int bit;
            stringstream line(buffer);
            while (line >> bit) {
                // 将第index行的被消元行对应位 置1
                shared_E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
            }
            index++;
        }
    }

    // 使用 MPI 的时间计时
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 同步共享内存窗口
    MPI_Win_lock_all(0, win_R);
    MPI_Win_lock_all(0, win_E);

    // 并行求解
    solve_parallel(rank, size, shared_R, shared_E);

    // 同步共享内存窗口
    MPI_Win_unlock_all(win_R);
    MPI_Win_unlock_all(win_E);

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 时间计时结束
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 打印结果
    if (rank == 0) {
        cout << elapsed_time * 1000 << "ms" << endl;
    }

    // 释放共享内存
    MPI_Win_free(&win_R);
    MPI_Win_free(&win_E);

    MPI_Finalize();

    return 0;
}
