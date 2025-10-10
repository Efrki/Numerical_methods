#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using matrix_data = std::vector<std::vector<double>>;
using vector_data = std::vector<double>;

void print_vector(const vector_data &vec, const std::string &name = "x") {
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << name << i + 1 << " = " << vec[i] << std::endl;
  }
}

double vector_norm_inf(const vector_data &vec) {
  double max_abs = 0.0;
  for (const auto &value : vec) {
    max_abs = std::max(max_abs, std::abs(value));
  }
  return max_abs;
}

double matrix_norm_inf(const matrix_data &A) {
  double max_row_sum = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    double current_row_sum = 0.0;
    for (size_t j = 0; j < A[i].size(); ++j) {
      current_row_sum += std::abs(A[i][j]);
    }
    max_row_sum = std::max(max_row_sum, current_row_sum);
  }
  return max_row_sum;
}

vector_data calculate_residual(const matrix_data &A, const vector_data &x,
                               const vector_data &b) {
  int size = A.size();
  vector_data Ax(size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      Ax[i] += A[i][j] * x[j];
    }
  }

  vector_data residual(size);
  for (int i = 0; i < size; ++i) {
    residual[i] = b[i] - Ax[i];
  }
  return residual;
}

bool is_diagonally_dominant(const matrix_data &A) {
  for (size_t i = 0; i < A.size(); ++i) {
    double diag_element = std::abs(A[i][i]);
    double sum_of_row = 0.0;
    for (size_t j = 0; j < A.size(); ++j) {
      if (i != j) {
        sum_of_row += std::abs(A[i][j]);
      }
    }
    if (diag_element < sum_of_row) {
      std::cerr
          << "Предупреждение: Нарушено диагональное преобладание в строке "
          << i + 1 << "." << std::endl;
      std::cerr << "Диагональный элемент: " << diag_element
                << ", Сумма остальных: " << sum_of_row << std::endl;
      return false;
    }
  }
  return true;
}

vector_data solve_jacobi(const matrix_data &A, const vector_data &b,
                         double epsilon, int &iterations) {
  int size = A.size();
  vector_data x(size, 0.0);
  vector_data x_new(size, 0.0);
  iterations = 0;
  const int max_iterations = 1000;

  while (iterations < max_iterations) {
    for (int i = 0; i < size; ++i) {
      double sum = 0.0;
      for (int j = 0; j < size; ++j) {
        if (i != j) {
          sum += A[i][j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i][i];
    }

    double max_diff = 0.0;
    for (int i = 0; i < size; ++i) {
      max_diff = std::max(max_diff, std::abs(x_new[i] - x[i]));
    }

    x = x_new;
    iterations++;

    if (max_diff < epsilon) {
      return x;
    }
  }

  std::cerr << "Метод Якоби не сошелся за " << max_iterations << " итераций."
            << std::endl;
  return x;
}

vector_data solve_seidel(const matrix_data &A, const vector_data &b,
                         double epsilon, int &iterations) {
  int size = A.size();
  vector_data x(size, 0.0);
  iterations = 0;
  const int max_iterations = 1000;

  while (iterations < max_iterations) {
    vector_data x_old = x;

    for (int i = 0; i < size; ++i) {
      double sum1 = 0.0;
      for (int j = 0; j < i; ++j) {
        sum1 += A[i][j] * x[j];
      }
      double sum2 = 0.0;
      for (int j = i + 1; j < size; ++j) {
        sum2 += A[i][j] * x_old[j];
      }
      x[i] = (b[i] - sum1 - sum2) / A[i][i];
    }

    double max_diff = 0.0;
    for (int i = 0; i < size; ++i) {
      max_diff = std::max(max_diff, std::abs(x[i] - x_old[i]));
    }

    iterations++;

    if (max_diff < epsilon) {
      return x;
    }
  }

  std::cerr << "Метод Зейделя не сошелся за " << max_iterations << " итераций."
            << std::endl;
  return x;
}

int main() {
  int size;
  std::cout << "Введите размерность матрицы: ";
  std::cin >> size;

  if (size <= 0) {
    std::cerr << "Ошибка: размер матрицы должен быть положительным."
              << std::endl;
    return 1;
  }

  matrix_data A(size, vector_data(size));
  vector_data b(size);
  double epsilon;

  std::cout << "Введите " << size * size
            << " элементов матрицы A построчно:" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << "Строка " << i + 1 << ": ";
    for (int j = 0; j < size; ++j) {
      std::cin >> A[i][j];
    }
  }

  std::cout << "\nВведите " << size << " элементов вектора b:" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cin >> b[i];
  }

  std::cout << "\nВведите точность вычислений (например, 0.0001): ";
  std::cin >> epsilon;

  std::cout << "\n--- Проверка на диагональное преобладание ---" << std::endl;
  if (is_diagonally_dominant(A)) {
    std::cout << "Матрица обладает диагональным преобладанием. Сходимость "
                 "гарантирована."
              << std::endl;
  } else {
    std::cout << "Предупреждение: матрица не обладает диагональным "
                 "преобладанием. Сходимость не гарантирована."
              << std::endl;
  }

  std::cout << "\n--- Норма матрицы ---" << std::endl;
  std::cout << "Норма матрицы A: " << matrix_norm_inf(A) << std::endl;

  std::cout << "\n--- Метод простых итераций (Якоби) ---" << std::endl;
  int jacobi_iterations = 0;
  vector_data x_jacobi = solve_jacobi(A, b, epsilon, jacobi_iterations);
  std::cout << "Решение найдено за " << jacobi_iterations
            << " итераций:" << std::endl;
  print_vector(x_jacobi, "x");

  std::cout << "\n--- Метод Зейделя ---" << std::endl;
  int seidel_iterations = 0;
  vector_data x_seidel = solve_seidel(A, b, epsilon, seidel_iterations);
  std::cout << "Решение найдено за " << seidel_iterations
            << " итераций:" << std::endl;
  print_vector(x_seidel, "x");

  vector_data residual_jacobi = calculate_residual(A, x_jacobi, b);
  double norm_jacobi = vector_norm_inf(residual_jacobi);

  vector_data residual_seidel = calculate_residual(A, x_seidel, b);
  double norm_seidel = vector_norm_inf(residual_seidel);

  std::cout << "\n--- Анализ количества итераций ---" << std::endl;
  std::cout << "Метод Якоби: " << jacobi_iterations << " итераций."
            << std::endl;
  std::cout << "Метод Зейделя: " << seidel_iterations << " итераций."
            << std::endl;

  if (seidel_iterations < jacobi_iterations) {
    std::cout << "Метод Зейделя сошелся быстрее." << std::endl;
  } else if (jacobi_iterations < seidel_iterations) {
    std::cout << "Метод Якоби сошелся быстрее." << std::endl;
  } else {
    std::cout << "Оба метода потребовали одинаковое количество итераций."
              << std::endl;
  }

  return 0;
}