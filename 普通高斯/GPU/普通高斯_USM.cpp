%%writefile lab/matrix.cpp
#include <iostream>
#include <random>
#include <chrono>
#include <sycl/sycl.hpp>

using namespace sycl;

static const int N = 4096;

// 生成随机矩阵数据
void generateRandomData(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = dis(gen);
        }
    }
}

// 串行代码
void LU(float* mat, int n) {
    float* mat_device = (float*)malloc(sizeof(float) * n * n);
    memcpy(mat_device, mat, sizeof(float) * n * n);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            mat_device[k * n + j] = mat_device[k * n + j] / mat_device[k * n + k];
        }
        mat_device[k * n + k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                mat_device[i * n + j] = mat_device[i * n + j] - mat_device[k * n + j] * mat_device[i * n + k];
            }
            mat_device[i * n + k] = 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double time_used = duration.count();
    std::cout << "串行算法用时: " << time_used << std::endl;

    if (n > 16) {
        free(mat_device);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << mat_device[i * n + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(mat_device);
}

// 并行代码
void LU_gpu(float* mat, int n) {
    // 顺序执行
    queue q{ gpu_selector_v };

    std::cout << "并行算法使用设备: " << q.get_device().get_info<info::device::name>() << std::endl;

    float* mat_device = (float*)malloc_shared<float>(n * n, q);
    std::memcpy(mat_device, mat, sizeof(float) * n * n);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        q.submit([&](handler& h) {
            h.parallel_for(range(n - (k + 1)), [=](auto idx) {
                int j = k + 1 + idx;
                mat_device[k * n + j] /= mat_device[k * n + k];
            });
        }).wait();

        mat_device[k * n + k] = 1;

        q.submit([&](handler& h) {
            // 从第 k + 1 行开始消去，消去包括第 k 列
            h.parallel_for(range(n - (k + 1), n - k), [=](auto idx) {
                int i = k + 1 + idx.get_id(0);
                int j = k + idx.get_id(1);
                mat_device[i * n + j] -= mat_device[i * n + k] * mat_device[k * n + j];
            });
        }).wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double time_used = duration.count();
    std::cout << "并行算法用时: " << time_used << std::endl;

    if (n > 16) {
        free(mat_device, q);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << mat_device[i * n + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
    free(mat_device, q);
}

int main() {
    float* mat = new float[N * N];
    generateRandomData(mat, N);

    std::vector<int> problemSizes = {200, 500, 1000, 2000, 3000};
    for (int i = 0; i < 5; i++) {
        int n = problemSizes[i];  // 系数矩阵的大小
        std::cout << "矩阵大小：" << n << " * " << n << std::endl;
        LU(mat, n);
        LU_gpu(mat, n);
    }

    delete[] mat;

    return 0;
}
