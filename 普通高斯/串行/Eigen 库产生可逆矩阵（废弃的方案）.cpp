#include <iostream>
#include <Eigen/Dense>

int main() {
    int n = 3; // 矩阵的维度
    
    // 创建一个n×n的随机矩阵
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(n, n);
    
    // 检查矩阵是否可逆
    while (matrix.determinant() == 0) {
        // 如果矩阵不可逆，则重新生成随机矩阵
        matrix = Eigen::MatrixXd::Random(n, n);
    }
    
    std::cout << "可逆矩阵为：\n" << matrix << "\n";
    
    return 0;
}
