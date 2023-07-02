#include <mpi.h>
#include <iostream>
#include <vector>
#include "../random.h"

using namespace std;

void GaussianEliminationMPI(float** A, int n, int myid, int r1, int r2, int num) {
    MPI_Request* send_requests = new MPI_Request[num];
    MPI_Request* recv_request = new MPI_Request;
    MPI_Status* statuses = new MPI_Status[num];

    for (int k = 0; k < n; k++) {
        int request_count = 0;
        if (r1 <= k && k <= r2) {
            for (int j = k + 1; j < n; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;

            for (int j = 0; j < num; j++) {
                if (j != myid) {
                    MPI_Isend(&A[k][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &send_requests[request_count]);
                    request_count++;
                }
            }
        } else {
            // 其他进程在消去前需要接收消息
            MPI_Irecv(&A[k][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, recv_request);
            MPI_Wait(recv_request, MPI_STATUS_IGNORE);
        }

        for (int i = r1; i <= r2; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[k][j] * A[i][k];
                }
                A[i][k] = 0;
            }
        }

        // 非阻塞通信结束
        if (r1 <= k && k <= r2) {
            MPI_Waitall(request_count, send_requests, statuses);
        }
    }

    delete[] send_requests;
    delete recv_request;
    delete[] statuses;
}

int main(int argc, char** argv) {
    int myid, numprocs;
    vector<int> problemSizes = {200, 500, 1000, 2000, 3000};
    MPI_Init(&argc, &argv);
    for (int i=0; i<5; i++) {
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
        float** A = new float*[n];
        for (int i = 0; i < n; i++) {
            A[i] = new float[n];
        }
        reset(A, n);

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
        for (int i = 0; i < n; i++) {
            delete[] A[i];
        }
        delete[] A;
    }
    MPI_Finalize();
    return 0;
}
