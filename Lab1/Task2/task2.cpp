#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vector>

int main(){
    int size = 0;

    std::cout << "Введите порядок матрицы: ";
    std::cin >> size;

    if (size < 2) {
        std::cout << "Ошибка: неправильный размер матрицы";
        return -1;
    }

    std::cout << std::endl;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> c(size);
    std::vector<float> d(size);

    std::cout << "Теперь введите коэффициенты для " << size << " уравнений" << std::endl;
    std::cout << "Для i-го уравнения введите 4 числа: a_i, b_i, c_i, d_i" << std::endl;
    std::cout << "Примечание: для первого уравнения a_1 = 0, а для последнего c_n = 0" << std::endl << std::endl;

    for (int i = 0; i < size; ++i) {
        std::cout << "Введите коэффициенты для " << i + 1 << "-го уравнения: ";
        std::cin >> a[i] >> b[i] >> c[i] >> d[i];
        std::cout << std::endl;
    }

    std::cout << "Проверим устойчивость для таких коэффициентов..." << std::endl;
    for (int i = 0; i < size; ++i) {
        if (abs(b[i]) < abs(a[i] + abs(c[i]))) {
            std::cout << "Ошибка: неправильные коэффициенты!";
            return -1;
        }
    }

    std::cout << "Всё устойчиво! Продолжаем..." << std::endl;

    std::vector<float> Ps(size);
    std::vector<float> Qs(size);

    Ps[0] = -c[0] / b[0];
    Qs[0] = d[0] / b[0];

    for (int j = 1; j < size; ++j) {
        Ps[j] = -c[j] / (b[j] + a[j] * Ps[j - 1]);
        Qs[j] = (d[j] - a[j] * Qs[j - 1]) / (b[j] + a[j] * Ps[j - 1]);
    }

    std::vector<float> Xs(size);

    Xs[size - 1] = Qs[size - 1];

    for (int i = size - 2; i >= 0; --i) {
        Xs[i] = Ps[i] * Xs[i + 1] + Qs[i];
    }

    for (int i = 0; i < size; ++i) {
        std::cout << "x " << i + 1 << "-e равняется: " << Xs[i] << std::endl;
    }

    return 0;
}
