#include <iostream>
#include <vector>

struct matrix {
    int size;
    std::vector<std::vector<int>> data;

    matrix(int s) : size(s), data(s, std::vector<int>(s)) {}

    void fill_from_input() {
        std::cout << "Введите " << size * size << " элементов матрицы построчно:" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << "Строка " << i + 1 << ": ";
            for (int j = 0; j < size; ++j) {
                std::cin >> data[i][j];
            }
        }
    }

    void print() const {
        std::cout << "\nСодержимое матрицы:" << std::endl;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                std::cout << data[i][j] << "\t"; 
            }
            std::cout << std::endl; 
        }
    }
};

matrix create_unit_matrix(int size){
    matrix Mat(size);

    for (int i = 0; i < size; ++i) {
        Mat.data[i][i] = 1;
    }

    return Mat;
}

int main(){
    int size = 0;

    std::cout << "Введите размерность матрицы: ";
    std::cin >> size;
    std::cout << std::endl;

    if (size < 2) {
        std::cout << "Ошибка: неверный размер матрицы"<<std::endl;
        return -1;
    }

    matrix A(size);

    A.fill_from_input();

    matrix U(size), L(size), P(size);
    U = A;
    L = create_unit_matrix(size);
    P = create_unit_matrix(size);


    return 0;
}