#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace std;

// 生成0到1之间的浮点随机数
float get_random_float() {
    static mt19937 gen(42);  // 取种子42，方便对比调试
    uniform_real_distribution<> dis(0.0, 1.0);  // 均匀分布
    return dis(gen);
}

void LU(vector<vector<float>>& A, int n) {
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
    vector<int> size = { 200, 500, 1000, 2000, 3000 };
    for (int n : size) {
        // 初始化二维数组并生成随机数
        vector<vector<float>> A(n, vector<float>(n, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = get_random_float();
            }
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
