#include <mpi.h>
#include <iostream>
#include <vector>
#include "../random.h"

using namespace std;

void GaussianEliminationMPI(float* A, int n, int myid, int r1, int r2, int num) {
    for (int k = 0; k < n; k++) {
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < n; j++) {
                A[k * n + j] = A[k * n + j] / A[k * n + k];
            }
            A[k * n + k] = 1.0;

            for (int j = 0; j < num; j++) {
                if (j != myid) {
                    MPI_Send(&A[k * n], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(&A[k * n], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = r1; i <= r2; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i * n + j] -= A[k * n + j] * A[i * n + k];
                }
                A[i * n + k] = 0;
            }
        }
    }
}

int main(int argc, char** argv) {
    int myid, numprocs;
    vector<int> problemSizes = {200, 500, 1000, 2000, 3000};
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
            r2 += remainder; // 最后一个进程负责的行数可能会多一些
        }

        // 创建并初始化系数矩阵
        float* A = new float[n * n];
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = dis(gen); // 初始化 (i, j) 处的矩阵元素
            }
        }

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        GaussianEliminationMPI(A, n, myid, r1, r2, numprocs);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        if (myid == 0) {
            cout << "Size = " << n << ": " << diff.count() << "ms" << endl;
        }

        // // 打印结果
        // if (myid == 0 && n == 1000) {
        //     print(A, n);
        // }

        // 释放内存
        delete[] A;
    }
    MPI_Finalize();
    return 0;
}
