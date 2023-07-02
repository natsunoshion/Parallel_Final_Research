#include <bits/stdc++.h>

using namespace std;

// 对应于数据集三个参数：矩阵列数，非零消元子行数，被消元行行数
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// 数组长度
const int length = ceil(num_columns / 5.0);

// 使用char数组进行存储，R：消元子，E：被消元行
char R[10000][length];  // R[i]记录了首项为i（下标从0开始记录）的消元子行
                        // 所以不能直接用num_elimination_rows设置数组大小
char E[num_eliminated_rows][length];

// 记录是否升格
bool lifted[num_eliminated_rows];

// 将当前位图打印到屏幕上
void print() {
    for (int i=0; i<num_eliminated_rows; i++) {
        // cout << i << ':';
        for (int j=num_columns-1; j>=0; j--) {
            // 第j位为1
            if ((((E[i][j / 5] >> (j - 5*(j/5)))) & 1) == 1) {
                cout << j << ' ';
            }
        }
        // for (int j=length-1; j>=0; j--) {
        //     cout << E[i][j] << ' ';
        // }
        cout << endl;
    }
}

// 特殊高斯消去法串行int数组实现
void solve() {
    // 遍历被消元行：逐元素消去，一次清空
    for (int i = 0; i < num_eliminated_rows; i++) {
        bool is_eliminated = false;
        // 行内遍历依次找首项消去
        for (int j = length - 1; j >= 0; j--) {
            // cout << j << "消去" << endl;
            for (int k=4; k>=0; k--) {
                if ((E[i][j] >> k) == 1) {
                    // 获得首项
                    int leader = 5 * j + k;
                    if (R[leader][j] != 0) {
                        // 有消元子就消去，需要对整行异或
                        for (int m=j; m>=0; m--) {
                            E[i][m] ^= R[leader][m];
                        }
                    } else {
                        // 否则升格，升格之后这一整行都可以不用管了
                        for (int m=j; m>=0; m--) {
                            R[leader][m] = E[i][m];
                        }
                        // 跳出多重循环
                        is_eliminated = true;
                    }
                }
                if (is_eliminated) {
                    break;
                }
            }
            if (is_eliminated) {
                break;
            }
        }
    }
}

int main() {
    // 读入消元子
    ifstream file_R;
    char buffer[10000] = {0};
    // file_R.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/消元子.txt");
    file_R.open("/home/data/Groebner/4_1011_539_263/1.txt");
    if (file_R.fail()) {
        cout << "读入失败" << endl;
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        int first_in = 1;

        // 消元子的index是其首项
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }

            // 将第index行的消元子对应位 置1
            R[index][bit / 5] |= 1 << (bit - (bit / 5) * 5);
        }
    }
    file_R.close();
//--------------------------------
    // 读入被消元行
    ifstream file_E;
    // file_E.open("/home/data/Groebner/测试样例1 矩阵列数130，非零消元子22，被消元行8/被消元行.txt");
    file_E.open("/home/data/Groebner/4_1011_539_263/2.txt");

    // 被消元行的index就是读入的行数
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // 每一次读入一行，消元子每32位记录进一个int中
        int bit;
        stringstream line(buffer);
        while (line >> bit) {
            // 将第index行的被消元行对应位 置1
            E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
        }
        index++;
    }
    // cout << E[6][50];
    // print();
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