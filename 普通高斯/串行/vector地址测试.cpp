#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec(5);
    int arr[5];

    for (int i = 0; i < 5; i++) {
        vec[i] = i;
        arr[i] = i;
    }

    std::cout << "Vector element addresses:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << &vec[i] << std::endl;
    }

    std::cout << "Array element addresses:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << &arr[i] << std::endl;
    }

    return 0;
}
