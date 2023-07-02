#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "../random.h"  // 使用自制的随机库

using namespace std;

// 数据块类型，表示要处理的矩阵的一部分
struct DataBlock {
    float** A;
    int n;
    int start;
    int end;
};

// 任务队列类
class TaskQueue {
private:
    queue<DataBlock> tasks;
    mutex mtx;
    condition_variable cv;

public:
    void enqueue(const DataBlock& task) {
        lock_guard<mutex> lock(mtx);
        tasks.push(task);
        cv.notify_one();
    }

    DataBlock dequeue() {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this] { return !tasks.empty(); });
        DataBlock task = tasks.front();
        tasks.pop();
        return task;
    }
};

// 并行执行任务的线程函数
void workerFunction(TaskQueue& taskQueue) {
    while (true) {
        DataBlock task = taskQueue.dequeue();

        // 处理任务
        for (int k = task.start; k < task.end; k++) {
            for (int j = k + 1; j < task.n; j++) {
                task.A[k][j] /= task.A[k][k];
            }
            task.A[k][k] = 1.0;
            for (int i = k + 1; i < task.n; i++) {
                for (int j = k + 1; j < task.n; j++) {
                    task.A[i][j] -= task.A[k][j] * task.A[i][k];
                }
                task.A[i][k] = 0;
            }
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

        const int numThreads = thread::hardware_concurrency();
        vector<thread> threads;
        TaskQueue taskQueue;

        // 创建线程池并启动线程
        for (int t = 0; t < numThreads; t++) {
            threads.emplace_back(workerFunction, ref(taskQueue));
        }

        // 将任务划分为块并加入任务队列
        int chunkSize = n / numThreads;
        for (int t = 0; t < numThreads; t++) {
            int start = t * chunkSize;
            int end = (t == numThreads - 1) ? n : (start + chunkSize);
            taskQueue.enqueue({A, n, start, end});
        }

        // 等待所有线程完成任务
        for (auto& t : threads) {
            t.join();
        }

        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
        cout << "Size = " << n << ": " << diff.count() << "ms" << endl;

        // 释放二维数组A的空间
        for (int i = 0; i < n; i++) {
            delete[] A[i];
        }
        delete[] A;
    }
    return 0;
}
