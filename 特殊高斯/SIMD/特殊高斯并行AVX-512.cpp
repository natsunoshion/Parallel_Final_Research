#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

// MAXM:最大行数，MAXN：列数
const int MAXM = 10000;
const int MAXN = 8399;

// 使用bitset进行存储，R：消元子，E：被消元行
bitset<MAXN> R[MAXM];
bitset<MAXN> E[MAXM];

// 被消元行行数
int m = 4535;

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
            R_temp[i] = bitset<5>((R[i].to_ulong()<<(MAXN-n-1)) >> (MAXN-5));
            E_temp[i] = bitset<5>((E[i].to_ulong()<<(MAXN-n-1)) >> (MAXN-5));
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
        // 每16段一组进行并行化，不断从records中取列
        for (auto pair : records) {
            int row = pair.first;
            int leader = pair.second;
            int m;
            for (m=n; m>=79; m-=80) {
                // 被消元行
                bitset<5> a1_bit = bitset<5>(((E[row]<<(MAXN-m-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a2_bit = bitset<5>(((E[row]<<(MAXN-(m-5)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a3_bit = bitset<5>(((E[row]<<(MAXN-(m-10)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a4_bit = bitset<5>(((E[row]<<(MAXN-(m-15)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a5_bit = bitset<5>(((E[row]<<(MAXN-(m-20)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a6_bit = bitset<5>(((E[row]<<(MAXN-(m-25)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a7_bit = bitset<5>(((E[row]<<(MAXN-(m-30)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a8_bit = bitset<5>(((E[row]<<(MAXN-(m-35)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a9_bit = bitset<5>(((E[row]<<(MAXN-(m-40)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a10_bit = bitset<5>(((E[row]<<(MAXN-(m-45)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a11_bit = bitset<5>(((E[row]<<(MAXN-(m-50)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a12_bit = bitset<5>(((E[row]<<(MAXN-(m-55)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a13_bit = bitset<5>(((E[row]<<(MAXN-(m-60)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a14_bit = bitset<5>(((E[row]<<(MAXN-(m-65)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a15_bit = bitset<5>(((E[row]<<(MAXN-(m-70)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> a16_bit = bitset<5>(((E[row]<<(MAXN-(m-75)-1)) >> (MAXN-5)).to_ulong());
                // 消元子
                bitset<5> b1_bit = bitset<5>(((R[leader]<<(MAXN-m-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b2_bit = bitset<5>(((R[leader]<<(MAXN-(m-5)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b3_bit = bitset<5>(((R[leader]<<(MAXN-(m-10)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b4_bit = bitset<5>(((R[leader]<<(MAXN-(m-15)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b5_bit = bitset<5>(((R[leader]<<(MAXN-(m-20)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b6_bit = bitset<5>(((R[leader]<<(MAXN-(m-25)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b7_bit = bitset<5>(((R[leader]<<(MAXN-(m-30)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b8_bit = bitset<5>(((R[leader]<<(MAXN-(m-35)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b9_bit = bitset<5>(((R[leader]<<(MAXN-(m-40)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b10_bit = bitset<5>(((R[leader]<<(MAXN-(m-45)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b11_bit = bitset<5>(((R[leader]<<(MAXN-(m-50)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b12_bit = bitset<5>(((R[leader]<<(MAXN-(m-55)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b13_bit = bitset<5>(((R[leader]<<(MAXN-(m-60)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b14_bit = bitset<5>(((R[leader]<<(MAXN-(m-65)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b15_bit = bitset<5>(((R[leader]<<(MAXN-(m-70)-1)) >> (MAXN-5)).to_ulong());
                bitset<5> b16_bit = bitset<5>(((R[leader]<<(MAXN-(m-75)-1)) >> (MAXN-5)).to_ulong());
                // 形成整数，构成neon向量
                int a1 = a1_bit.to_ulong();
                int a2 = a2_bit.to_ulong();
                int a3 = a3_bit.to_ulong();
                int a4 = a4_bit.to_ulong();
                int a5 = a5_bit.to_ulong();
                int a6 = a6_bit.to_ulong();
                int a7 = a7_bit.to_ulong();
                int a8 = a8_bit.to_ulong();
                int a9 = a9_bit.to_ulong();
                int a10 = a10_bit.to_ulong();
                int a11 = a11_bit.to_ulong();
                int a12 = a12_bit.to_ulong();
                int a13 = a13_bit.to_ulong();
                int a14 = a14_bit.to_ulong();
                int a15 = a15_bit.to_ulong();
                int a16 = a16_bit.to_ulong();
                int b1 = b1_bit.to_ulong();
                int b2 = b2_bit.to_ulong();
                int b3 = b3_bit.to_ulong();
                int b4 = b4_bit.to_ulong();
                int b5 = b5_bit.to_ulong();
                int b6 = b6_bit.to_ulong();
                int b7 = b7_bit.to_ulong();
                int b8 = b8_bit.to_ulong();
                int b9 = b9_bit.to_ulong();
                int b10 = b10_bit.to_ulong();
                int b11 = b11_bit.to_ulong();
                int b12 = b12_bit.to_ulong();
                int b13 = b13_bit.to_ulong();
                int b14 = b14_bit.to_ulong();
                int b15 = b15_bit.to_ulong();
                int b16 = b16_bit.to_ulong();
                int arr_a[16] = {a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16};
                int arr_b[16] = {b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16};
                __m512i_u va = _mm512_set_epi32(a16, a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1);
                __m512i_u vb = _mm512_set_epi32(b16, b15, b14, b13, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1);
                va = _mm512_xor_si512(va, vb);  // 异或
                _mm512_storeu_si512((__m256i*)arr_a, va);  // 存回去
                // 存储回被消元行的位图，按arr_a[i]的5个位进行置位操作
                // 外层循环遍历arr_a数组，内存循环遍历arr_a[i]的位数
                for (int i=0; i<16; i++) {
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
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("R.txt");
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
    return 0;
}