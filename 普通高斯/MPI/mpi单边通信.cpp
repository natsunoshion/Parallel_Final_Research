#include <mpi.h>
#include <iostream>
#include <vector>
#include "../random.h"

using namespace std;

void GaussianEliminationMPI(float** A, int n, int myid, int r1, int r2, int num) {
    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        // 每个进程只需要处理自己负责的行进行消去
        for (int i = r1; i <= r2; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }
    }
}

int main(int argc, char** argv) {
    int myid, numprocs;
    vector<int> problemSizes = {200, 500, 1000, 2000, 3000};
    float** A;
    MPI_Win win;
    MPI_Init(&argc, &argv);
    for (int i = 0; i < 5; i++) {
        int n = problemSizes[i];  // 系数矩阵的大小
        int r1, r2;   // 负责的起始行和终止行

        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        // 计算每个进程负责的行范围
        int rows_per_process = n / numprocs;
        int remainder = n % numprocs;

        r1 = myid * rows_per_process;
        r2 = r1 + rows_per_process - 1;

        if (myid == numprocs - 1) {
            r2 += remainder; // 最后一个进程可能负责额外的行数
        }

        A = new float*[n];
        for (int j = 0; j < n; j++) {
            A[j] = new float[n];
        }

        // 创建共享矩阵
        MPI_Win_allocate(sizeof(float*) * n, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &A, &win);

        // 0 号进程初始化
        if (myid == 0) {
            reset(A, n);
        }

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        GaussianEliminationMPI(A, n, myid, r1, r2, numprocs);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        if (myid == 0) {
            cout << "Size = " << n << ": " << diff.count() << "ms" << endl;
        }

        // 打印结果
        if (myid == 0 && n == 200) {
            print(A, n);
        }

        for (int j = 0; j < n; j++) {
            delete[] A[j];
        }
        delete[] A;

        // 释放内存
        MPI_Win_free(&win);
    }
    MPI_Finalize();
    return 0;
}
