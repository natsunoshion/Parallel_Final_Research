#include <iostream>
#include <chrono>
#include "../random.h"
#include <omp.h>

using namespace std;

#define NUM_THREADS 4

// 都放静态存储区，节省内存
float** A;
int n;

// OpenMP高斯消去，LU分解
void LU() {
    // 循环外创建线程，避免线程反复创建销毁，影响程序性能
    float tmp;
    int i, j, k;
    bool parallel = true;
    // print(A, N);
    #pragma omp target teams distribute parallel for if(parallel) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) private(i, j, k, tmp)
    for (k=0; k<n; k++) {
        // 串行除法
        #pragma omp single
        {
            tmp = A[k][k];
            // 由于用了tmp，所以A[k][k]也可以合在循环变量中了
            for (j=k; j<n; j++) {
                A[k][j] /= tmp;
            }
        }
        // print(A, N);
        // cout << endl;

        // 使用OpenMP并行化行消去
        // 使用OpemMP 4.0的SIMD方法进行自动向量化、循环展开
        #pragma omp target teams distribute parallel for simd
        // #pragma omp target teams distribute parallel for
        for (i=k+1; i<n; i++) {
            tmp = A[i][k];
            for (j=k; j<n; j++) {
                A[i][j] -= A[k][j] * tmp;
            }
        }
    }
}

int main() {
    vector<int> size = {200, 500, 1000, 2000, 3000};
    for (int i=0; i<5; i++) {
        // 设置问题规模
        n = size[i];

        // 初始化二维数组
        A = new float*[n];
        for (int i=0; i<n; i++) {
            A[i] = new float[n];
        }
        reset(A, n);

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        LU();
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << n << ": " << diff.count() << "ms" << endl;

        // print(A, n);
        // break;

        // 释放二维数组A的空间
        for (int i = 0; i < n; i++) {
            delete[] A[i];
        }
        delete[] A;
    }
    return 0;
}