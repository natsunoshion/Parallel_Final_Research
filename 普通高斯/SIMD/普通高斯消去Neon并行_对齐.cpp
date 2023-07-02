#include <iostream>
#include <chrono>
#include <arm_neon.h>  // Neon
#include "../random.h"

using namespace std;

// 普通高斯消元Neon并行算法
void LU(float** A, int N) {
    for (int k=0; k<N; k++) {
        // 并行化本行除法
        // float dup[4] = {A[k][k], A[k][k], A[k][k], A[k][k]};
        // vt的向量寄存器为：{A[k][k], A[k][k], A[k][k], A[k][k]}
        float32x4_t vt = vld1q_dup_f32(&A[k][k]);  // 复制4份A[k][k]存进vt中
        int j = k + 1;

        // A[k][j]需要内存对齐
        while (j % 4 != 0) {
            A[k][j] = A[k][j] / A[k][k];
            j++;
        }
        for (; j+4<=N; j+=4) {
            // va的向量寄存器为：{A[k][j], A[k][j+1], A[k][j+2], A[k][j+3]}
            float32x4_t va = vld1q_f32(&A[k][j]);  // Neon下的对齐指令

            // va = va / vt
            va = vdivq_f32(va, vt);

            // 将va寄存器存储到原位置，完成4个数的除法
            vst1q_f32(&A[k][j], va);
        }

        // 处理剩下的元素，由于最多不超过3个，所以直接串行除法就可以了
        for (; j<N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i=k+1; i<N; i++) {
            // 对第i行的元素计算进行并行化处理
            // A[i][j]、A[k][j]开始连续四个元素分别形成寄存器
            // A[i][k]为固定值，复制4份存在另一个寄存器里
            float32x4_t vaik = vld1q_dup_f32(&A[i][k]);
            int j = k + 1;

            // A[k][j]、A[i][j]需要内存对齐
            while (j % 4 != 0) {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            for (; j+4<=N; j+=4) {
                // 原始公式：A[i][j] = A[i][j] - A[k][j]*A[i][k];
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }

            // 剩下的元素
            for (; j<N; j++) {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    int N;
    vector<int> size = {200, 500, 1000, 2000, 3000};
    for (int i=0; i<5; i++) {
        // 设置问题规模
        N = size[i];

        // 初始化二维数组
        float** A = new float*[N];
        for (int i=0; i<N; i++) {
            A[i] = new float[N];
        }

        // 使用随机数重置数组
        reset(A, N);

        // 使用C++11的chrono库来计时
        auto start = chrono::high_resolution_clock::now();
        LU(A, N);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << N << ": " << diff.count() << "ms" << endl;
    }
    return 0;
}