#include <iostream>
#include <chrono>
#include <random>
#include <vector>

using namespace std;

// 生成0到1之间的浮点随机数
float get_random_float() {
    static mt19937 gen(42);  // 取种子42，方便对比调试
    uniform_real_distribution<> dis(0.0, 1.0);  // 均匀分布
    return dis(gen);
}

void LU(vector<float>& A, int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A[k * n + j] /= A[k * n + k];
        }
        A[k * n + k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0;
        }
    }
}

int main() {
    vector<int> size = { 200, 500, 1000, 2000, 3000 };
    for (int n : size) {
        // 初始化一维vector并生成随机数
        vector<float> A(n * n);
        for (int j = 0; j < n * n; j++) {
            A[j] = get_random_float();
        }

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        LU(A, n);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << n << ": " << diff.count() << "ms" << endl;
    }
    return 0;
}
