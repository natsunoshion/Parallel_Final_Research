#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

#define NUM_THREADS 4

const int length = 203;

char R[10000][length]; // R矩阵
char E[num_eliminated_rows][length]; // E矩阵

vector<pair<int, int>> records; // 保存记录的向量
bool lifted[num_eliminated_rows]; // 表示行是否已被消元

mutex mtx_leader; // 互斥锁 - 主线程
mutex mtx_division[NUM_THREADS - 1]; // 互斥锁 - 分割点同步
mutex mtx_elimination[NUM_THREADS - 1]; // 互斥锁 - 消元同步

void print() {
    // 打印结果
    for (int i = 0; i < num_eliminated_rows; i++) {
        for (int j = num_columns - 1; j >= 0; j--) {
            if (((E[i][j / 5] >> (j - 5 * (j / 5)))) & 1 == 1) {
                cout << j << ' ';
            }
        }
        cout << endl;
    }
}

void thread_func(int id) {
    // 线程函数
    for (int n = length - 1; n >= 0; n--) {
        if (id == 0) {
            records.clear();
            // 消元过程
            for (int i = 0; i < num_eliminated_rows; i++) {
                if (lifted[i]) {
                    continue;
                }
                for (int j = 5 * (n + 1) - 1; j >= 5 * n; j--) {
                    if (E[i][n] >> (j - 5 * n) == 1) {
                        if (R[j][n] != 0) {
                            E[i][n] ^= R[j][n];
                            records.emplace_back(i, j);
                        } else {
                            for (auto pair : records) {
                                int row = pair.first;
                                int leader = pair.second;
                                if (row == i) {
                                    for (int k = n - 1; k >= 0; k--) {
                                        E[i][k] ^= R[leader][k];
                                    }
                                }
                            }
                            for (int k = 0; k < length; k++) {
                                R[j][k] = E[i][k];
                            }
                            lifted[i] = true;
                            break;
                        }
                    }
                }
            }
            // 解锁分割点同步互斥锁
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                mtx_division[i].unlock();
            }
        } else {
            mtx_division[id - 1].lock(); // 加锁，等待分割点同步
        }
        // 执行消元操作
        for (int i = id; i < n; i += NUM_THREADS) {
            for (auto pair : records) {
                int row = pair.first;
                int leader = pair.second;
                if (lifted[row]) {
                    continue;
                }
                E[row][i] ^= R[leader][i];
            }
        }
        if (id == 0) {
            // 解锁主线程互斥锁
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                mtx_leader.lock();
            }
            // 解锁消元同步互斥锁
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                mtx_elimination[i].unlock();
            }
        } else {
            mtx_leader.unlock(); // 解锁主线程互斥锁
            mtx_elimination[id - 1].lock(); // 加锁，等待消元同步
        }
    }
}

void solve() {
    vector<thread> threads;
    // 创建线程
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(thread_func, i);
    }
    // 等待线程结束
    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    ifstream file_R;
    char buffer[10000] = {0};
    // 读取R矩阵文件
    file_R.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/消元子.txt");
    if (file_R.fail()) {
        cout << "读入失败" << endl;
        return 1;
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        int bit;
        stringstream line(buffer);
        int first_in = 1;
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }
            R[index][bit / 5] |= 1 << (bit - (bit / 5) * 5);
        }
    }
    file_R.close();

    ifstream file_E;
    // 读取E矩阵文件
    file_E.open("D:/study/vscode/parallel/Parallel_Final_Research/特殊高斯/Groebner/测试样例4 矩阵列数1011，非零消元子539，被消元行263/被消元行.txt");
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        int bit;
        stringstream line(buffer);
        while (line >> bit) {
            E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
        }
        index++;
    }
    file_E.close();

    solve();

    return 0;
}
