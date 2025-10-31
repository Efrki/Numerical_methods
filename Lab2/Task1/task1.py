import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return math.log(1 + x) - 2 * x**2 + 1

def f_prime(x):
    return 1.0 / (1.0 + x) - 4.0 * x

def f_double_prime(x):
    return -1.0 / ((1.0 + x)**2) - 4.0

def phi(x):
    if 1 + x <= 0:
        raise ValueError("Аргумент логарифма должен быть > 0 (1+x > 0)")
    arg_sqrt = (math.log(1 + x) + 1.0) / 2.0
    if arg_sqrt < 0:
        raise ValueError("Аргумент квадратного корня должен быть >= 0")
    return math.sqrt(arg_sqrt)

# Производная функции phi(x) для проверки условия сходимости
def phi_prime(x):
    if 1 + x <= 0:
        raise ValueError("Аргумент логарифма должен быть > 0 (1+x > 0)")
    arg_sqrt = (math.log(1 + x) + 1.0) / 2.0
    if arg_sqrt <= 0:
        raise ValueError("Аргумент квадратного корня должен быть > 0")
    denominator = 4 * (1 + x) * math.sqrt(arg_sqrt)
    return 1.0 / denominator

def solve_simple_iteration(x0, epsilon, max_iterations=100):
    print("\n--- Метод простой итерации ---")
    print(f"Начальное приближение x0 = {x0}")
    print(f"{'Iter':<5}{'x_i':<20}{'x_{i+1}':<20}{'|x_{i+1} - x_i|':<20}")
    print("-" * 65)

    # Проверка достаточного условия сходимости
    L = abs(phi_prime(x0))
    print(f"Проверка условия сходимости: |phi'({x0})| = {L:.4f}")
    if L >= 1:
        print("Предупреждение: Условие сходимости |phi'(x0)| < 1 не выполнено. Сходимость не гарантирована.")

    x_current = x0
    iterations = 0
    x_history = [x0]

    while iterations < max_iterations:
        try:
            x_next = phi(x_current)
        except ValueError as e:
            print(f"Ошибка в функции phi(x) на итерации {iterations + 1} (x={x_current:.10f}): {e}")
            break

        diff = abs(x_next - x_current)

        print(f"{iterations + 1:<5}{x_current:<20.10f}{x_next:<20.10f}{diff:<20.10f}")

        if diff < epsilon:
            print("\nРешение найдено!")
            print(f"Корень x = {x_next:.10f}")
            print(f"Количество итераций: {iterations + 1}")
            
            residual = f(x_next)
            print(f"\nПроверка решения подстановкой в f(x):")
            print(f"f({x_next:.10f}) = {residual:.3e}")
            if abs(residual) < epsilon:
                print(f"Успех: |f(x)| < epsilon ({abs(residual):.3e} < {epsilon:.3e})")
            else:
                print(f"Предупреждение: |f(x)| >= epsilon ({abs(residual):.3e} >= {epsilon:.3e})")

            x_history.append(x_next)
            return x_next, x_history

        x_current = x_next
        x_history.append(x_current)
        iterations += 1

    print(f"\nМетод простой итерации не сошелся за {max_iterations} итераций.")
    return None, x_history

def solve_newton(x0, epsilon, max_iterations=100):
    print("\n--- Метод Ньютона ---")
    print(f"Начальное приближение x0 = {x0}")
    print(f"{'Iter':<5}{'x_i':<20}{'f(x_i)':<20}{'f\'(x_i)':<20}{'|x_{i+1} - x_i|':<20}")
    print("-" * 85)

    if f(x0) * f_double_prime(x0) <= 0:
        print(f"Предупреждение: Условие сходимости f(x0)*f''(x0) > 0 не выполнено.")
        print(f"f({x0}) = {f(x0):.4f}, f''({x0}) = {f_double_prime(x0):.4f}")
        print("Сходимость не гарантирована.")
        
    x_current = x0
    iterations = 0
    x_history = [x0]

    while iterations < max_iterations:
        try:
            fx = f(x_current)
            fpx = f_prime(x_current)
        except ValueError as e:
            print(f"Ошибка в функции f(x) или f_prime(x) на итерации {iterations + 1} (x={x_current:.10f}): {e}")
            break

        if abs(fpx) < 1e-12:
            print("Ошибка: производная близка к нулю. Метод Ньютона не может продолжаться.")
            break

        x_next = x_current - fx / fpx
        diff = abs(x_next - x_current)

        print(f"{iterations + 1:<5}{x_current:<20.10f}{fx:<20.10e}{fpx:<20.10e}{diff:<20.10f}")

        if diff < epsilon:
            print("\nРешение найдено!")
            print(f"Корень x = {x_next:.10f}")
            print(f"Количество итераций: {iterations + 1}")

            residual = f(x_next)
            print(f"\nПроверка решения подстановкой в f(x):")
            print(f"f({x_next:.10f}) = {residual:.3e}")
            if abs(residual) < epsilon:
                print(f"Успех: |f(x)| < epsilon ({abs(residual):.3e} < {epsilon:.3e})")
            else:
                print(f"Предупреждение: |f(x)| >= epsilon ({abs(residual):.3e} >= {epsilon:.3e})")

            x_history.append(x_next)
            return x_next, x_history

        x_current = x_next
        x_history.append(x_current)
        iterations += 1

    print(f"\nМетод Ньютона не сошелся за {max_iterations} итераций.")
    return None, x_history

def plot_results(f_func, x_simple_iter_history, x_newton_history, root_simple_iter, root_newton):
    all_x_points = []
    if x_simple_iter_history:
        all_x_points.extend(x_simple_iter_history)
    if x_newton_history:
        all_x_points.extend(x_newton_history)
    if root_simple_iter is not None:
        all_x_points.append(root_simple_iter)
    if root_newton is not None:
        all_x_points.append(root_newton)

    min_x = min(all_x_points) - 0.1 if all_x_points else 0.5
    max_x = max(all_x_points) + 0.1 if all_x_points else 1.5
    
    min_x = max(min_x, -0.9)

    x_vals = np.linspace(min_x, max_x, 400)
    y_vals = [f_func(x) for x in x_vals]

    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, label='f(x) = ln(1+x) - 2x^2 + 1', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

    if x_simple_iter_history:
        x_si = np.array(x_simple_iter_history)
        y_si = np.array([f_func(x) for x in x_si])
        plt.plot(x_si, y_si, 'o-', color='red', label='Метод простой итерации (x_i, f(x_i))', alpha=0.7)
        plt.plot(x_si, np.zeros_like(x_si), 'x', color='red', alpha=0.7, markersize=8, label='Метод простой итерации (x_i, 0)')

    if x_newton_history:
        x_n = np.array(x_newton_history)
        y_n = np.array([f_func(x) for x in x_n])
        plt.plot(x_n, y_n, 's-', color='green', label='Метод Ньютона (x_i, f(x_i))', alpha=0.7)
        plt.plot(x_n, np.zeros_like(x_n), '+', color='green', alpha=0.7, markersize=8, label='Метод Ньютона (x_i, 0)')

    plt.title('Решение нелинейного уравнения: ln(1+x) - 2x^2 + 1 = 0')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    epsilon = float(input("Введите точность вычислений (например, 1e-6): "))
    x0 = 0.9

    root_si, history_si = solve_simple_iteration(x0, epsilon)
    root_newton, history_newton = solve_newton(x0, epsilon)

    plot_results(f, history_si, history_newton, root_si, root_newton)
