#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include "../random.h"  // 使用自制的随机库

using namespace std;

// 第 k 行向下消去
void LU(float** A, int n, int k, int start, int end) {
    for (int i = start; i < end; i++) {
        for (int j = k + 1; j < n; j++) {
            A[i][j] -= A[k][j] * A[i][k];
        }
        A[i][k] = 0;
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

        // 并行化处理
        int num_threads = thread::hardware_concurrency();
        cout << "线程数为: " << num_threads << endl;

        for (int k = 0; k < n; k++) {
            // 串行完成除法操作
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;

            vector<thread> threads;
            
            // 按块划分
            int chunk_size = (n - k) / num_threads;
            for (int t = 0; t < num_threads; t++) {
                int start = k + 1 + t * chunk_size;
                int end = (t == num_threads - 1) ? n : (start + chunk_size);
                threads.emplace_back([&]() { LU(A, n, k, start, end); });
            }

            // 等待所有线程完成
            for (auto& t : threads) {
                t.join();
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << n << ": " << diff.count() << "ms" << endl;

        print(A, n);
        break;

        // 释放二维数组A的空间
        for (int i = 0; i < n; i++) {
            delete[] A[i];
        }
        delete[] A;
    }
    return 0;
}
