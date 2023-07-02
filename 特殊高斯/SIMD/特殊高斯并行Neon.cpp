#include <bits/stdc++.h>
#include <arm_neon.h>

using namespace std;

// MAXM:最大行数，MAXN：列数
const int MAXM = 10000;
const int MAXN = 3799;

// 使用bitset进行存储，R：消元子，E：被消元行
bitset<MAXN> R[MAXM];
bitset<MAXN> E[MAXM];

// 被消元行行数
int m = 1953;

// 找到当前被消元行的首项
int lp(bitset<5> temp) {
    int k = 4;
    while (temp[k]==0 && k>=0) {
        k--;
    }
    return k;
}

// 特殊高斯消去法并行Neon实现，假设每一轮取5列消元子/被消元行出来
void solve() {
    int n = MAXN - 1;
    while (n >= 4) {
        // 每一轮计算n~n-4这5列的消元子，也就是min_R = n-4, max_R = n
        // 使用位运算，先左移MAXN-n-1位让所需要的5位变为最高位，然后再右移MAXN-5位，清除掉低位
        bitset<5> R_temp[MAXM];
        bitset<5> E_temp[MAXM];
        for (int i=0; i<MAXM; i++) {
            R_temp[i] = bitset<5>(((R[i]<<(MAXN-n-1)) >> (MAXN-5)).to_ulong());
            E_temp[i] = bitset<5>(((E[i]<<(MAXN-n-1)) >> (MAXN-5)).to_ulong());
        }
        vector<pair<int, int>> records;
        // 记录这5列的消元操作
        // 遍历被消元行
        for (int i=0; i<5; i++) {
            bool is_eliminated = 0;
            // 消元，并记录操作，这里记录使用vector<pair<int, int>>存储，第一个int记录被消元行操作的行号，第二个int记录首项所在的列号
            // 遍历消元子的行
            for (int j=4; j>=0; j--) {
                if (lp(E_temp[i]) == j) {
                    E_temp[i] ^= R_temp[j];
                    is_eliminated = 1;
                    records.emplace_back(i, j+n-4);  // 记录
                }
            }
            if (!is_eliminated) {
                // 不为空行则升格，为空行则舍去
                if (!E_temp[i].none()) {
                    R[lp(E_temp[i])] = E[i];
                }
                break;
            }
        }
        // 接下来，对这n列进行并行计算，按照records中的记录进行多线程操作（由于刚刚没有存回去，所以这里剩下有n列）
        // 每4段一组进行并行化，不断从records中取列
        for (auto pair : records) {
            int row = pair.first;
            int leader = pair.second;
            int m;
            for (m=n; m>=19; m-=20) {
                // 被消元行
                bitset<5> a1_bit = bitset<5>(((E[row]<<(MAXN-m-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a2_bit = bitset<5>(((E[row]<<(MAXN-(m-5)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a3_bit = bitset<5>(((E[row]<<(MAXN-(m-10)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a4_bit = bitset<5>(((E[row]<<(MAXN-(m-15)-1)) >> (MAXN-5)).to_ulong());
                // 消元子
                bitset<5> b1_bit = bitset<5>(((R[leader]<<(MAXN-m-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b2_bit = bitset<5>(((R[leader]<<(MAXN-(m-5)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b3_bit = bitset<5>(((R[leader]<<(MAXN-(m-10)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b4_bit = bitset<5>(((R[leader]<<(MAXN-(m-15)-1)) >> (MAXN-5)).to_ulong());
                // 形成整数，构成neon向量
                uint a1 = a1_bit.to_ulong();
                uint a2 = a2_bit.to_ulong();
                uint a3 = a3_bit.to_ulong();
                uint a4 = a4_bit.to_ulong();
                uint b1 = b1_bit.to_ulong();
                uint b2 = b2_bit.to_ulong();
                uint b3 = b3_bit.to_ulong();
                uint b4 = b4_bit.to_ulong();
                uint arr_a[4] = {a1, a2, a3, a4};
                uint arr_b[4] = {b1, b2, b3, b4};
                uint32x4_t va = vld1q_u32(arr_a);
                uint32x4_t vb = vld1q_u32(arr_b);
                va = veorq_u32(va, vb);  // 异或
                vst1q_u32(arr_a, va);
                // 存储回被消元行的位图，按arr_a[i]的5个位进行置位操作
                // 外层循环遍历arr_a数组，内存循环遍历arr_a[i]的位数
                for (int i=0; i<4; i++) {
                    for (int j=0; j<5; j++) {
                        E[row].set(m-j, arr_a[i] & 0x1);
                        arr_a[i] >>= 1;
                    }
                }
            }
            // 剩下的直接一个一个异或就可以了，使用掩码
            for (; m>=0; m--) {
                std::bitset<MAXN> mask;
                mask.reset();
                mask.set(m, 1);
                E[row] ^= (R[leader]&mask);
            }
        }
        n -= 5;
    }
    // 最后，n不到5了，也不用记录了
    bitset<5> R_temp[MAXM];
    bitset<5> E_temp[MAXM];
    for (int i=0; i<MAXM; i++) {
        R_temp[i] = bitset<5>(((R[i]<<(MAXN-n-1)) >> (MAXN-n-1)).to_ulong());
        E_temp[i] = bitset<5>(((E[i]<<(MAXN-n-1)) >> (MAXN-n-1)).to_ulong());
    }
    // 遍历被消元行
    for (int i=0; i<5; i++) {
        bool is_eliminated = 0;
        // 消元，并记录操作，这里记录使用vector<pair<int, int>>存储，第一个int记录被消元行操作的行号，第二个int记录首项所在的列号
        // 遍历消元子的行
        for (int j=4; j>=0; j--) {
            if (lp(E_temp[i]) == j) {
                E_temp[i] ^= R_temp[j];
                is_eliminated = 1;
            }
        }
        if (!is_eliminated) {
            // 不为空行则升格，为空行则舍去
            if (!E_temp[i].none()) {
                R[lp(E_temp[i])] = E[i];
            }
            break;
        }
    }
    // 存储回去
    for (int i=0; i<MAXM; i++) {
        for (int j=0; j<n+1; j++) {
            E[i].set(j, E_temp[i][j]);
        }
    }
}

int main() {
    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    file_R.open("/home/data/Groebner/6_3799_2759_1953/1.txt");
    // file_R.open("R.txt");
    if (file_R.fail()) {
        cout << "wow" << endl;
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
    file_E.open("/home/data/Groebner/6_3799_2759_1953/2.txt");
    // file_E.open("E.txt");

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
    return 0;
}