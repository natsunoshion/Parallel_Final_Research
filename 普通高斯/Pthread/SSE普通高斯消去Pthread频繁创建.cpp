#include <iostream>
#include <chrono>
#include "../random.h"
#include <pthread.h>  // Pthread编程
#include <nmmintrin.h>

using namespace std;

// 都放静态存储区，节省内存
float** A;
int n;

// 用于线程函数传参：线程号id和当前行k
typedef struct {
    int id;
    int k;
} ThreadArgs;

// 并行函数
void* thread_func(void* arg) {
    // 传参
    ThreadArgs* thread_arg = (ThreadArgs*)arg;
    int id = thread_arg->id;
    int k = thread_arg->k;

    // 一个线程负责一行
    int i = k + id + 1;
    // 创建向量
    __m128 vx, vaij, vaik, vakj;
    // 对第i行的元素计算进行并行化处理
    // A[i][j]、A[k][j]开始连续四个元素分别形成寄存器
    // A[i][k]为固定值，复制4份存在另一个寄存器里
    vaik = _mm_load1_ps(&A[i][k]);
    int j;
    for (j=k+1; j+4<=n; j+=4) {
        // 原始公式：A[i][j] = A[i][j] - A[k][j]*A[i][k];
        vakj = _mm_loadu_ps(&A[k][j]);
        vaij = _mm_loadu_ps(&A[i][j]);
        vx = _mm_mul_ps(vakj, vaik);
        vaij = _mm_sub_ps(vaij, vx);
        _mm_storeu_ps(&A[i][j], vaij);
    }

    // 剩下的元素
    for (; j<n; j++) {
        A[i][j] = A[i][j] - A[k][j]*A[i][k];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
}

// Pthread动态线程
void LU() {
    for (int k=0; k<n; k++) {
        // 串行除法
        for (int i=k; i<n; i++) {
            A[k][i] /= A[k][k];
        }
        // 多线程并行减法，此方法缺点：频繁创建线程，开销大
        // 初始化线程的id
        int worker_count = n - 1 - k;
        pthread_t* threads = new pthread_t[n - 1 - k];
        ThreadArgs* thread_ids = new ThreadArgs[n - 1 - k];
        for (int i=0; i<n-1-k; i++) {
            thread_ids[i] = {i, k};
        }
        // 创建线程
        for (int i=0; i<n-1-k; i++) {
            pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
        }
        // 等待线程计算完毕
        for (int i=0; i<n-1-k; i++) {
            pthread_join(threads[i], NULL);
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

        // 使用随机数重置数组
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