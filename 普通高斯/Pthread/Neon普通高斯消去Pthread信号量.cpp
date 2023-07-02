#include <iostream>
#include <chrono>
#include "../random.h"
#include <pthread.h>  // Pthread编程
#include <semaphore.h>  // 使用信号量
#include <arm_neon.h>

using namespace std;

#define NUM_THREADS 4

// 都放静态存储区，节省内存
float** A;
int n;

// 信号量，每个线程都有两个信号量，分别记录这一轮工作的开始和结束
sem_t sem_start[NUM_THREADS], sem_end[NUM_THREADS];

// 用于线程函数传参：线程号id
// 由于只创建一次线程，所以不需要记录当前行k，而是一个类似主线程与工作线程“轮流”运算的操作，两边的循环变量k是一起变的
typedef struct {
    int id;
} ThreadArgs;

// 并行函数
// 每个线程都拥有一个这个函数，只是id不同，所需要处理的部分就不同了
void* thread_func(void* arg) {
    // 在循环外创建向量
    float32x4_t vx, vaij, vaik, vakj;
    // 传参
    ThreadArgs* thread_arg = (ThreadArgs*)arg;
    int id = thread_arg->id;

    for (int k=0; k<n; k++) {
        sem_wait(&sem_start[id]);
        // 消去第[k+1, n)行的第k列元素
        // 按行间隔划分，交给NUM_THREADS个线程来处理
        for (int i = k+1+id; i<n; i += NUM_THREADS) {
            vaik = vld1q_dup_f32(&A[i][k]);
            int j;
            // j: k ~ n-1，向量化
            for (j=k; j+4<=n; j+=4) {
                // A[i][j] -= A[i][k] * A[k][j];
                vakj = vld1q_f32(&A[k][j]);
                vaij = vld1q_f32(&A[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            // 不能整除的部分
            for (; j<n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
        // 告诉主线程这个线程结束
        sem_post(&sem_end[id]);
    }
    pthread_exit(NULL);
}

// 普通高斯消去，LU分解
void LU() {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_ids[NUM_THREADS];

    // 初始化信号量
    for (int i=0; i<NUM_THREADS; i++) {
        sem_init(&sem_start[i], 0, 0);
        sem_init(&sem_end[i], 0, 0);
    }

    // 先在循环外创建线程
    // 初始化线程的id
    for (int i=0; i<NUM_THREADS; i++) {
        thread_ids[i] = {i};
    }
    for (int i=0; i<NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
    }

    for (int k=0; k<n; k++) {
        // 串行除法
        for (int i=k; i<n; i++) {
            A[k][i] /= A[k][k];
        }
        // 唤醒工作线程
        for (int i=0; i<NUM_THREADS; i++) {
            sem_post(&sem_start[i]);
        }
        // 主线程等待工作线程并行减法运算
        for (int i=0; i<NUM_THREADS; i++) {
            sem_wait(&sem_end[i]);
        }
    }

    // 线程工作结束
    for (int i=0; i<NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
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