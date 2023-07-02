#include <bits/stdc++.h>

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 254
#define num_elimination_rows 106
#define num_eliminated_rows 53

// 使用BitMagic进行存储，R：消元子，E：被消元行
bitset<num_columns> R[10000];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                         // 所以不能直接用num_elimination_rows设置数组大小
bitset<num_columns> E[num_eliminated_rows];

// 记录是否升格
bool lifted[num_eliminated_rows];

// 找到当前被消元行的首项
int lp(bitset<num_columns> temp) {
    // 获取最高位（最左边）为1的位置
    int pos = num_columns - 1;
    while (pos >= 0) {
        if (temp.test(pos)) {
            return pos;
        }
        pos--;
    }
    return pos;
}

// 对位范围进行异或操作
void block_xor(bitset<num_columns>& a, bitset<num_columns>& b, int start, int end) {
    for (int i = start; i <= end; i++) {
        // 按位异或
        bool bit_a = a[i];
        bool bit_b = b[i];
        bool result = bit_a ^ bit_b;
        a.set(i, result);
    }
}

// 测试start~end是否有1
bool any_range(bitset<num_columns> a, int start, int end) {
    for (int i = start; i <= end; i++) {
        if (a[i] == 1) {
            return true;
        }
    }
    return false;
}

// 特殊高斯消去法并行OpenMP实现，假设每一轮取5列消元子/被消元行出来
void solve() {
    int n;
    for (n = num_columns - 1; n >= 4; n -= 5) {
        // 每一轮计算n~n-4这5列的消元子，也就是min_R = n-4, max_R = n
        // 使用BitMagic库，运算就方便许多了
        // 对被消元行n~n-4这5列消元，并记录消元操作
        vector<pair<int, int>> records;  // 记录 <消元行操作的行号，首项所在的列号>
        for (int i=0; i<num_eliminated_rows; i++) {
            if (lifted[i]) {
                continue;
            }
            // 找首项
            for (int j=n; j>=n-4; j--) {
                // cout << "第" << i << "行" << n << '~' << n-4 << "列首项" << lp(E_temp[i]) << endl;
                if (lp(E[i]) == j) {
                    if (R[j].any()) {
                        block_xor(E[i], R[j], n-4, n);
                        // 记录(行号，消元子首项)
                        records.emplace_back(i, j);
                    }
                }
            }
            // 没有消干净的话，必须立刻升格，方便其他行消去
            if (any_range(E[i], n-4, n)) {
                for (auto pair : records) {
                    int row = pair.first;
                    int leader = pair.second;
                    if (row == i) {
                        // 补上剩下位的异或
                        block_xor(E[i], R[leader], 0, n-5);
                    }
                }
                // 删除这些对被消元行第i行操作的记录
                records.erase(std::remove_if(records.begin(), records.end(), [i](const std::pair<int, int>& p) { return p.first == i; }), records.end());
                // 升格
                R[lp(E[i])] = E[i];
                lifted[i] = true;
            }
        }
        // 接下来，对后n-4列进行并行计算，按照records中的记录进行多线程操作（由于刚刚没有存回去，所以这里剩下有n列）
        // OpenMP多线程并行化
        #pragma omp parallel for simd schedule(guided, 1)
        for (int m=n-5; m>=0; m-=5) {
            for (auto pair : records) {
                int row = pair.first;
                int leader = pair.second;
                // 使用max合并边界情况
                block_xor(E[row], R[leader], max(m-4, 0), m);
            }
        }
    }
    // 最后，n不到5了，直接遍历消去E，不用记录
    // 遍历被消元行
    for (int i=0; i<num_eliminated_rows; i++) {
        if (lifted[i]) {
            continue;
        }
        for (int j=n; j>=0; j--) {
            if (lp(E[i]) == j) {
                if (R[j].any()) {
                    E[i] ^= R[j];
                }
            }
        }
        // 不为空行则升格
        if (E[i].any()) {
            R[lp(E[i])] = E[i];
            lifted[i] = true;
        }
    }
}

void print() {
    for (int i=0; i<num_eliminated_rows; i++) {
        // cout << i << ':';
        for (int j=num_columns-1; j>=0; j--) {
            if (E[i][j] == 1) {
                cout << j << ' ';
            }
        }
        cout << endl;
    }
}

int main() {
    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("R.txt");
    if (file_R.fail()) {
        cout << "读入失败" << endl;
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        int first_in = 1;

        // 消元子的索引是其首项
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }

            // 将第index行的消元子bitset对应位 置1
            R[index][bit] = 1;
        }
    }
    file_R.close();
//--------------------------------
    // 读入被消元行
    ifstream file_E;
    // file_E.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/被消元行.txt");
    file_E.open("E.txt");

    // 被消元行的索引就是读入的行数
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        while (line >> bit) {
            // 将第index行的消元子bitset对应位 置1
            E[index][bit] = 1;
        }
        index++;
    }
//--------------------------------
    // 使用C++11的chrono库来计时
    auto start = chrono::high_resolution_clock::now();
    solve();
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << diff.count() << "ms" << endl;
//--------------------------------
    // 验证结果正确性
    print();
    return 0;
}