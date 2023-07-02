#include <iostream>
#include <chrono>
#include <immintrin.h>  // AVX、AVX2
#include "../random.h"

using namespace std;

// 普通高斯消元AVX-512并行算法
void LU(float** A, int N) {
    for (int k=0; k<N; k++) {
        // 并行化本行除法
        // 这里我在AVX-512指令集中没有找到类似于SSE中_mm_load1_ps这样方便的指令，所以使用新建数组的方式来复制16份A[k][k]存进vt中
        float dup[16] = {A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]};

        // vt的向量寄存器为：{A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]}
        __m512 vt = _mm512_loadu_ps(dup);
        int j = k + 1;

        // A[k][j]需要内存对齐，本来是要求(k*N + j) % 16 == 0，为了方便，这里N取16的倍数，条件就变为了j % 16 == 0
        while (j % 16 != 0) {
            A[k][j] = A[k][j] / A[k][k];
            j++;
        }
        for (; j+16<=N; j+=16) {
            // va的向量寄存器为：{A[k][j], A[k][j+1], A[k][j+2], A[k][j+3], A[k][j+4], A[k][j+5], A[k][j+6], A[k][j+7], A[k][j+8], A[k][j+9], A[k][j+10], A[k][j+11], A[k][j+12], A[k][j+13], A[k][j+14], A[k][j+15]}
            __m512 va = _mm512_loadu_ps(&A[k][j]);  // loadu，内存可以不对齐

            // va = va / vt
            va = _mm512_div_ps(va, vt);

            // 将va寄存器存储到原位置，完成16个数的除法
            _mm512_storeu_ps(&A[k][j], va);
        }

        // 处理剩下的元素
        for (; j<N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i=k+1; i<N; i++) {
            // 对第i行的元素计算进行并行化处理
            // A[i][j]、A[k][j]开始连续四个元素分别形成寄存器
            // A[i][k]为固定值，复制16份存在另一个寄存器里
            float dupik[16] = {A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]};
            __m512 vaik = _mm512_loadu_ps(dupik);
            int j = k + 1;

            // A[k][j]、A[i][j]需要内存对齐
            while (j % 16 != 0) {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            for (; j+16<=N; j+=16) {
                // 原始公式：A[i][j] = A[i][j] - A[k][j]*A[i][k];
                __m512 vakj = _mm512_loadu_ps(&A[k][j]);
                __m512 vaij = _mm512_loadu_ps(&A[i][j]);
                __m512 vx = _mm512_mul_ps(vakj, vaik);
                vaij = _mm512_sub_ps(vaij, vx);
                _mm512_storeu_ps(&A[i][j], vaij);
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