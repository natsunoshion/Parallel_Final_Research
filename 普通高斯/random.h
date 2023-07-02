#include <iostream>
#include <random>

using namespace std;

// static mt19937 gen(time(nullptr));  // 以时间作为种子
static mt19937 gen(42);  // 取种子 42，方便对比调试

// 生成0到1之间的浮点随机数
float get_random_float() {
    uniform_real_distribution<> dis(0.0, 1.0);  // 均匀分布
    return dis(gen);
}

void print(float** m, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << m[i][j] << ' ';
        }
        cout << endl;
    }
}

// 重新产生随机数矩阵
// 尝试过几天，事实上全部 random 才是最佳选择
void reset(float** m, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            m[i][j] = get_random_float();
        }
    }
}
