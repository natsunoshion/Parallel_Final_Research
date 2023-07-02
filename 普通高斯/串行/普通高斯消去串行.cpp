#include <iostream>
#include <chrono>
#include "../random.h"  // 使用自制的随机库

using namespace std;

void LU(float** A, int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    int size[] = { 200, 500, 1000, 2000, 3000 };
    for (int i = 0; i < 5; i++) {
        int n = size[i];

        // 初始化二维数组并生成随机数
        float** A = new float* [n];
        for (int i = 0; i < n; i++) {
            A[i] = new float[n];
        }

        // 重置数组
        reset(A, n);

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        LU(A, n);
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
