#include <iostream>
#include <chrono>
#include "../random.h"
#include <vector>
#include <omp.h>

using namespace std;

#define NUM_THREADS 4

// 缓存行大小，根据实际硬件确定
#define CACHE_LINE_SIZE 64

// 填充结构体
struct alignas(CACHE_LINE_SIZE) MyStruct {
    float data;
    char padding[CACHE_LINE_SIZE - sizeof(float)];  // 填充变量
};

// 使用的数据填充解决虚假共享问题
MyStruct** A;
int n;

// OpenMP高斯消去，LU分解
void LU() {
    float tmp;
    int i, j, k;
    bool parallel = true;

    #pragma omp parallel if(parallel) num_threads(NUM_THREADS) private(i, j, k, tmp)
    for (k = 0; k < n; k++) {
        #pragma omp single
        {
            tmp = A[k][k].data;
            for (j = k; j < n; j++) {
                A[k][j].data /= tmp;
            }
        }

        #pragma omp for simd
        for (i = k + 1; i < n; i++) {
            tmp = A[i][k].data;
            for (j = k; j < n; j++) {
                A[i][j].data -= A[k][j].data * tmp;
            }
        }
    }
}

int main() {
    vector<int> size = {200, 500, 1000, 2000, 3000};
    for (int i = 0; i < 5; i++) {
        n = size[i];

        A = new MyStruct*[n];
        for (int i = 0; i < n; i++) {
            A[i] = new MyStruct[n];
        }

        auto start = chrono::high_resolution_clock::now();
        LU();
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << n << ": " << diff.count() << "ms" << endl;

        for (int i = 0; i < n; i++) {
            delete[] A[i];
        }
        delete[] A;
    }
    return 0;
}