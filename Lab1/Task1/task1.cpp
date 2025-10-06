#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>

struct matrix {
  int size;
  std::vector<std::vector<double>> data;

  matrix(int s) : size(s), data(s, std::vector<double>(s)) {}

  void fill_from_input() {
    std::cout << "Введите " << size * size
              << " элементов матрицы построчно:" << std::endl;
    for (int i = 0; i < size; ++i) {
      std::cout << "Строка " << i + 1 << ": ";
      for (int j = 0; j < size; ++j) {
        std::cin >> data[i][j];
      }
    }
    std::cout << std::endl;
  }

  void print() {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        std::cout << data[i][j] << "\t";
      }
      std::cout << std::endl;
    }
  }

  std::vector<double> forward_substitution(const std::vector<double>& b) const {
      std::vector<double> y(size);
      for (int i = 0; i < size; ++i) {
          double sum = 0.0;
          for (int j = 0; j < i; ++j) {
              sum += data[i][j] * y[j];
          }
          y[i] = (b[i] - sum) / data[i][i];
      }
      return y;
  }

  std::vector<double> solve(const std::vector<double>& b) const {
        if (size != b.size()) {
            throw std::runtime_error("Matrix and vector size mismatch.");
        }

        std::vector<double> x(size);
        for (int i = size - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < size; ++j) {
                sum += data[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / data[i][i];
        }

        return x;
    }


};

matrix create_unit_matrix(int size) {
  matrix Mat(size);

  for (int i = 0; i < size; ++i) {
    Mat.data[i][i] = 1;
  }

  return Mat;
}

std::vector<double> multiply_matrix_vector(const matrix& P, const std::vector<double>& b) {
    std::vector<double> result(P.size);
    for (int i = 0; i < P.size; ++i) {
        result[i] = 0;
        for (int j = 0; j < P.size; ++j) {
            result[i] += P.data[i][j] * b[j];
        }
    }
    return result;
}

double determinant(const matrix& U, int permutations) {
    double det = (permutations % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < U.size; ++i) {
        det *= U.data[i][i];
    }
    return det;
}

int lup_decomposition(matrix& A, matrix& L, matrix& U, matrix& P) {
  int size = A.size;
  int permutations = 0;
  for (int k = 0; k < size - 1; ++k) {
    int pivot = k;
    double max_val = std::abs(U.data[k][k]);
    for (int i = k + 1; i < size; ++i) {
      if (std::abs(U.data[i][k]) > max_val) {
        max_val = std::abs(U.data[i][k]);
        pivot = i;
      }
    }

    if (pivot != k) {
      std::swap(U.data[k], U.data[pivot]);
      std::swap(P.data[k], P.data[pivot]);
      permutations++;
      for (int j = 0; j < k; ++j) {
        std::swap(L.data[k][j], L.data[pivot][j]);
      }
    }

    for (int i = k + 1; i < size; ++i) {
      double factor = (double)U.data[i][k] / U.data[k][k];
      L.data[i][k] = factor;
      for (int j = k; j < size; ++j) {
        U.data[i][j] -= factor * U.data[k][j];
      }
    }
  }
  return permutations;
}

matrix inverse(const matrix& L, const matrix& U, const matrix& P) {
    int n = L.size;
    for (int i = 0; i < n; ++i) {
        if (std::abs(U.data[i][i]) < 1e-12) {
            throw std::runtime_error("Матрица вырождена, невозможно найти обратную.");
        }
    }
    matrix inv(n);
    for (int i = 0; i < n; ++i) {
        std::vector<double> e(n, 0.0);
        e[i] = 1.0;

        std::vector<double> pe = multiply_matrix_vector(P, e);
        std::vector<double> y = L.forward_substitution(pe);
        std::vector<double> x = U.solve(y);

        for (int j = 0; j < n; ++j) {
            inv.data[j][i] = x[j];
        }
    }
    return inv;
}

std::vector<double> solve_slae(const matrix& L, const matrix& U, const matrix& P, const std::vector<double>& b) {
    std::vector<double> pb = multiply_matrix_vector(P, b);
    std::vector<double> y = L.forward_substitution(pb);

    std::vector<double> x = U.solve(y);
    return x;
}
matrix multiply(const matrix& A, const matrix& B) {
    if (A.size != B.size) {
        return matrix(0);
    }
    matrix answer(A.size);

    for (int i = 0; i < A.size; ++i) {
        for (int j = 0; j < A.size; ++j) {
            answer.data[i][j] = 0;
            for (int k = 0; k < A.size; ++k) {
                answer.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }

    return answer;
  }

int main() {
  int size = 0;

  std::cout << "Введите размерность матрицы: ";
  std::cin >> size;
  std::cout << std::endl;

  if (size < 2) {
    std::cout << "Ошибка: неверный размер матрицы" << std::endl;
    return -1;
  }

  matrix A(size);
  A.fill_from_input();

  matrix U = A;
  matrix L = create_unit_matrix(size);
  matrix P = create_unit_matrix(size);

  int permutations = lup_decomposition(A, L, U, P);

  std::cout << "L матрица:" << std::endl;
  L.print();
  std::cout << "U матрица:" << std::endl;
  U.print();
  std::cout << "P матрица:" << std::endl;
  P.print();

  std::cout << "\nDet(A): " << determinant(U, permutations) << std::endl;

  try {
    matrix A_inverse = inverse(L, U, P);
    std::cout << "\nобратная A:" << std::endl;
    A_inverse.print();

    matrix product = multiply(A, A_inverse);
    std::cout << "\nПроверка A * A_inverse:" << std::endl;
    product.print();

  bool is_identity = true;
  for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
          if (i == j && std::abs(product.data[i][j] - 1.0) > 1e-9) is_identity = false;
          if (i != j && std::abs(product.data[i][j]) > 1e-9) is_identity = false;
      }
  }
  if (is_identity) {
      std::cout << "\nПроизведение A * A_inverse является единичной матрицей (с учетом погрешностей)." << std::endl;
  } else {
      std::cout << "\nПроизведение A * A_inverse НЕ является единичной матрицей." << std::endl;
  }
  } catch (const std::runtime_error& e) {
      std::cerr << "\nОшибка при нахождении обратной матрицы: " << e.what() << std::endl;
  }

  std::vector<double> b(size);
  std::cout << "\nВведите коэффициенты матрицы b:" << std::endl;
  for (int i = 0; i < size; ++i) {
      std::cin >> b[i];
  }

  try {
      std::vector<double> x = solve_slae(L, U, P, b);
      std::cout << "\nРешение СЛАУ:" << std::endl;
      for (int i = 0; i < size; ++i) {
          std::cout << "x" << i + 1 << " = " << x[i] << std::endl;
      }

      std::cout << "\nПроверка решения (A*x):" << std::endl;
      std::vector<double> b_check = multiply_matrix_vector(A, x);
      for (int i = 0; i < size; ++i) {
          std::cout << "Строка " << i + 1 << ": A*x = " << b_check[i] 
                    << ", b = " << b[i] << std::endl;
      }
  } catch (const std::runtime_error& e) {
      std::cerr << "Ошибка решения СЛАУ: " << e.what() << std::endl;
  }

  return 0;
}