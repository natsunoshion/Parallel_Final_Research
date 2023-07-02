#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

// ��Ӧ�����ݼ���������������������������Ԫ������������Ԫ������
#define num_columns 1011
#define num_elimination_rows 539
#define num_eliminated_rows 263

// ���鳤��
const int length = 203;

// ʹ��char������д洢��R����Ԫ�ӣ�E������Ԫ��
char R[10000][length];  // R[i]��¼������Ϊi���±��0��ʼ��¼������Ԫ����
                        // ���Բ���ֱ����num_elimination_rows���������С
char E[num_eliminated_rows][length];

// ��¼�Ƿ�����
bool lifted[num_eliminated_rows];

// ����ǰλͼ��ӡ����Ļ��
void print() {
    for (int i = 0; i < num_eliminated_rows; i++) {
        // std::cout << i << ':';
        for (int j = num_columns - 1; j >= 0; j--) {
            // ��jλΪ1
            if ((((E[i][j / 5] >> (j - 5 * (j / 5)))) & 1) == 1) {
                std::cout << j << ' ';
            }
        }
        // for (int j = length - 1; j >= 0; j--) {
        //     std::cout << E[i][j] << ' ';
        // }
        std::cout << std::endl;
    }
}

// ����
void input() {
    // ������Ԫ��
    std::ifstream file_R;
    char buffer[10000] = {0};
    file_R.open("D:/study/vscode/parallel/Parallel_Final_Research/�����˹/Groebner/��������4 ��������1011��������Ԫ��539������Ԫ��263/��Ԫ��.txt");
    // file_R.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/R.txt");
    if (file_R.fail()) {
        std::cout << "Failed to open file" << std::endl;
        perror("File error");
        std::string command = "dir \"D:/study/vscode/parallel/Parallel_Final_Research/�����˹/Groebner/��������4 ��������1011��������Ԫ��539������Ԫ��263/��Ԫ��.txt\"";
        int result = std::system(command.c_str());
    }
    while (file_R.getline(buffer, sizeof(buffer))) {
        // ÿһ�ζ���һ�У���Ԫ��ÿ32λ��¼��һ��int��
        int bit;
        std::stringstream line(buffer);
        int first_in = 1;

        // ��Ԫ�ӵ�index��������
        int index;
        while (line >> bit) {
            if (first_in) {
                first_in = 0;
                index = bit;
            }

            // ����index�е���Ԫ�Ӷ�Ӧλ ��1
            R[index][bit / 5] |= 1 << (bit - (bit / 5) * 5);
        }
    }
    file_R.close();

    // ���뱻��Ԫ��
    std::ifstream file_E;
    file_E.open("D:/study/vscode/parallel/Parallel_Final_Research/�����˹/Groebner/��������4 ��������1011��������Ԫ��539������Ԫ��263/����Ԫ��.txt");
    // file_E.open("/home/u195976/oneAPI_Essentials/02_SYCL_Program_Structure/E.txt");

    // ����Ԫ�е�index���Ƕ��������
    int index = 0;
    while (file_E.getline(buffer, sizeof(buffer))) {
        // ÿһ�ζ���һ�У���Ԫ��ÿ32λ��¼��һ��int��
        int bit;
        std::stringstream line(buffer);
        while (line >> bit) {
            // ����index�еı���Ԫ�ж�Ӧλ ��1
            E[index][bit / 5] |= (1 << (bit - (bit / 5) * 5));
        }
        index++;
    }
}

// �����˹��ȥ������ char ����ʵ��
void solve() {
    auto start = std::chrono::high_resolution_clock::now();

    // ��������Ԫ�У���Ԫ����ȥ��һ�����
    for (int i = 0; i < num_eliminated_rows; i++) {
        bool is_eliminated = false;
        // ���ڱ���������������ȥ
        for (int j = length - 1; j >= 0; j--) {
            // cout << j << "��ȥ" << endl;
            for (int k = 4; k >= 0; k--) {
                if ((E[i][j] >> k) == 1) {
                    // �������
                    int leader = 5 * j + k;
                    if (R[leader][j] != 0) {
                        // ����Ԫ�Ӿ���ȥ����Ҫ���������
                        for (int m = j; m >= 0; m--) {
                            E[i][m] ^= R[leader][m];
                        }
                    } else {
                        // ������������֮����һ���ж����Բ��ù���
                        for (int m = j; m >= 0; m--) {
                            R[leader][m] = E[i][m];
                        }
                        // ��������ѭ��
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

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << diff.count() << "ms" << std::endl;
}

int main() {
    input();
    print();
    solve();
    // print();
    return 0;
}
