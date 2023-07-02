#include <iostream>
#include <chrono>
#include "../random.h"
#include <pthread.h>  // Pthread编程
#include <semaphore.h>  // 使用信号量

using namespace std;

#define NUM_THREADS 4

// 都放静态存储区，节省内存
float** A;
int n;

// 信号量,同步除法与消去
sem_t sem_leader;
sem_t sem_division[NUM_THREADS - 1];
sem_t sem_elimination[NUM_THREADS - 1];

// 用于线程函数传参：线程号id
// 由于只创建一次线程，所以不需要记录当前行k，而是一个类似主线程与工作线程“轮流”运算的操作，两边的循环变量k是一起变的
typedef struct {
    int id;
} ThreadArgs;

// 并行函数
// 三重循环全部放在线程函数中
// 每个线程都拥有一个这个函数，只是id不同，所需要处理的部分就不同了
void* thread_func(void* arg) {
    // 传参
    ThreadArgs* thread_arg = (ThreadArgs*)arg;
    int id = thread_arg->id;

    for (int k=0; k<n; k++) {
        // id为0的线程进行除法操作
        if (id == 0) {
            for (int j=k+1; j<n; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            // 除法完成,唤醒其他线程
            for (int i=0; i<NUM_THREADS-1; i++) {
                sem_post(&sem_division[i]);
            }
        } else {
            // 其余线程先等待
            sem_wait(&sem_division[id - 1]);
        }

        // 消去第[k+1, n)行的第k列元素
        // 按行间隔划分，交给NUM_THREADS个线程来处理
        // 第0号线程也会参与,提升线程利用率
        for (int i = k+1+id; i<n; i += NUM_THREADS) {
            for (int j=k+1; j<n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        // 以0号线程为基准同步
        if (id == 0) {
            // 等待其余工作线程结束消去操作
            for (int i=0; i<NUM_THREADS-1; i++) {
                sem_wait(&sem_leader);
            }
            // 通知消去完成
            for (int i=0; i<NUM_THREADS-1; i++) {
                sem_post(&sem_elimination[i]);
            }
        } else {
            // 通知leader
            sem_post(&sem_leader);
            // 等待0号线程的通知
            sem_wait(&sem_elimination[id - 1]);
        }
    }
    pthread_exit(NULL);
}

// 普通高斯消去，LU分解
void LU() {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_ids[NUM_THREADS];

    // 初始化线程的id
    for (int i=0; i<NUM_THREADS; i++) {
        thread_ids[i] = {i};
    }

    // 初始化信号量，这很重要
    sem_init(&sem_leader, 0, 0);
    for (int i=0; i<NUM_THREADS-1; i++) {
        sem_init(&sem_division[i], 0, 0);
        sem_init(&sem_elimination[i], 0, 0);
    }

    // 创建线程
    for (int i=0; i<NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
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