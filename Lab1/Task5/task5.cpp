#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <complex>

using matrix = std::vector<std::vector<double>>;
using vector = std::vector<double>;


void print_matrix(const matrix& A, const std::string& name) {
    std::cout << "Матрица " << name << ":" << std::endl;
    for (const auto& row : A) {
        for (const auto& val : row) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(6) << val;
        }
        std::cout << std::endl;
    }
}

matrix transpose(const matrix& A) {
    int rows = A.size();
    int cols = A[0].size();
    matrix T(cols, vector(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

matrix multiply(const matrix& A, const matrix& B) {
    int n = A.size();
    matrix C(n, vector(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


void qr_decomposition(const matrix& A, matrix& Q, matrix& R) {
    int n = A.size();
    matrix A_t = transpose(A);
    matrix Q_t(n, vector(n));

    for (int i = 0; i < n; ++i) {
        vector u = A_t[i];
        for (int j = 0; j < i; ++j) {
            double scalar_product = 0;
            for (int k = 0; k < n; ++k) {
                scalar_product += Q_t[j][k] * A_t[i][k];
            }
            for (int k = 0; k < n; ++k) {
                u[k] -= scalar_product * Q_t[j][k];
            }
        }

        double norm = 0;
        for (int k = 0; k < n; ++k) {
            norm += u[k] * u[k];
        }
        norm = std::sqrt(norm);

        for (int k = 0; k < n; ++k) {
            Q_t[i][k] = u[k] / norm;
        }
    }

    Q = transpose(Q_t);
    R = multiply(Q_t, A);
}


std::vector<std::complex<double>> qr_eigenvalues(matrix& A, double epsilon, int& iterations) {
    int n = A.size();
    iterations = 0;
    const int max_iterations = 2000;
    int current_size = n;

    while (current_size > 0 && iterations < max_iterations) {
        matrix Q(current_size, vector(current_size));
        matrix R(current_size, vector(current_size));
        matrix sub_A(current_size, vector(current_size));
        for(int i = 0; i < current_size; ++i) {
            for(int j = 0; j < current_size; ++j) {
                sub_A[i][j] = A[i][j];
            }
        }

        qr_decomposition(sub_A, Q, R);
        sub_A = multiply(R, Q);

        for(int i = 0; i < current_size; ++i) {
            for(int j = 0; j < current_size; ++j) {
                A[i][j] = sub_A[i][j];
            }
        }
        iterations++;
    }

    std::vector<std::complex<double>> eigenvalues;
    int i = 0;
    while (i < n) {
        if (i + 1 < n && std::abs(A[i+1][i]) > epsilon) {
            double a = A[i][i];
            double b = A[i][i+1];
            double c = A[i+1][i];
            double d = A[i+1][i+1];

            double trace = a + d;
            double det = a * d - b * c;

            double discriminant = trace * trace - 4 * det;

            if (discriminant < 0) {
                eigenvalues.emplace_back(trace / 2.0, std::sqrt(-discriminant) / 2.0);
                eigenvalues.emplace_back(trace / 2.0, -std::sqrt(-discriminant) / 2.0);
            } else {
                eigenvalues.emplace_back((trace + std::sqrt(discriminant)) / 2.0, 0);
                eigenvalues.emplace_back((trace - std::sqrt(discriminant)) / 2.0, 0);
            }
            i += 2;
        } else {
            eigenvalues.emplace_back(A[i][i], 0);
            i += 1;
        }
    }
    return eigenvalues;
}

int main() {
    int size;
    std::cout << "Введите размерность матрицы: ";
    std::cin >> size;

    if (size <= 0) {
        std::cerr << "Ошибка: размер матрицы должен быть положительным." << std::endl;
        return 1;
    }

    matrix A(size, vector(size));
    double epsilon;

    std::cout << "Введите " << size * size << " элементов матрицы A построчно:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << "Строка " << i + 1 << ": ";
        for (int j = 0; j < size; ++j) {
            std::cin >> A[i][j];
        }
    }

    std::cout << "\nВведите точность вычислений (например, 1e-6): ";
    std::cin >> epsilon;

    std::cout << "\n--- Исходная матрица ---" << std::endl;
    print_matrix(A, "A");

    int iterations = 0;
    std::vector<std::complex<double>> eigenvalues = qr_eigenvalues(A, epsilon, iterations);

    std::cout << "\n--- Результат QR-алгоритма ---" << std::endl;
    std::cout << "Выполнено " << iterations << " итераций." << std::endl;
    
    std::cout << "\nИтоговая матрица (вещественная форма Шура):" << std::endl;
    print_matrix(A, "A_final");

    std::cout << "\nНайденные собственные значения:" << std::endl;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        std::cout << "lambda_" << i + 1 << " = " << std::fixed << std::setprecision(6) << eigenvalues[i].real();
        if (std::abs(eigenvalues[i].imag()) > 1e-9) {
            if (eigenvalues[i].imag() > 0) {
                std::cout << " + " << eigenvalues[i].imag() << "i";
            } else {
                std::cout << " - " << -eigenvalues[i].imag() << "i";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
