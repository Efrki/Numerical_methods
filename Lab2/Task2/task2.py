import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, log

class NonlinearSystemSolver:
    def __init__(self, a=3):
        self.a = a
    
    def system(self, x):
        """Система уравнений: F(x) = 0"""
        x1, x2 = x
        f1 = x1**2 + x2**2 - self.a**2
        f2 = x1 - exp(x2) + self.a
        return np.array([f1, f2])
    
    def jacobian(self, x):
        """Матрица Якоби системы"""
        x1, x2 = x
        return np.array([
            [2*x1, 2*x2],
            [1, -exp(x2)]
        ])
    
    def phi_jacobian(self, x):
        """Матрица Якоби для итерационной функции Phi(X)"""
        x1, x2 = x
        # Защита от деления на ноль или корня из отрицательного числа
        sqrt_arg = self.a**2 - x2**2
        log_arg = x1 + self.a
        if sqrt_arg <= 0 or log_arg <= 0:
            # Возвращаем матрицу с большой нормой, чтобы показать, что условие не выполнено
            return np.array([[99, 99], [99, 99]])
        
        j12 = -x2 / sqrt(sqrt_arg)
        j21 = 1 / log_arg
        return np.array([[0, j12], [j21, 0]])

    def simple_iteration_method(self, x0, epsilon=1e-6, max_iter=1000):
        """
        Метод простой итерации
        x0 - начальное приближение
        epsilon - точность
        max_iter - максимальное число итераций
        """
        print("Метод простой итерации:")
        print(f"  Начальное приближение: {x0}")
        print(f"  Требуемая точность: {epsilon:.1e}")

        # Проверка достаточного условия сходимости
        J_phi = self.phi_jacobian(x0)
        L = np.linalg.norm(J_phi, ord=np.inf) # Используем бесконечную норму (max row sum)
        print(f"  Проверка условия сходимости: ||J_phi(x0)|| = {L:.4f}")
        if L >= 1:
            print("  Предупреждение: Условие сходимости ||J_phi(x0)|| < 1 не выполнено. Сходимость не гарантирована.")
        
        x = np.array(x0, dtype=float)
        errors = []
        iterations_count = []
        
        for i in range(max_iter):
            x_old = x.copy()
            
            x1, x2 = x
            x1_new = sqrt(max(0, self.a**2 - x2**2))
            x2_new = log(x1 + self.a) if (x1 + self.a) > 0 else x2
            
            x_new = np.array([x1_new, x2_new])
            x = x_new
            error = np.linalg.norm(x - x_old)
            errors.append(error)
            iterations_count.append(i + 1)
            
            print(f"  Итерация {i+1:<3}: x=[{x[0]:.8f}, {x[1]:.8f}], |dx|={error:.2e}")
            
            if error < epsilon:
                residual_norm = np.linalg.norm(self.system(x))
                print(f"\n  Сходимость достигнута за {i+1} итераций.")
                print(f"  Решение: x = [{x[0]:.10f}, {x[1]:.10f}]")
                print(f"  Норма невязки ||F(x)|| = {residual_norm:.2e}")
                return x, errors, iterations_count
        
        print(f"\n  Метод не сошелся за {max_iter} итераций.")
        residual_norm = np.linalg.norm(self.system(x))
        print(f"  Последнее приближение: x = [{x[0]:.10f}, {x[1]:.10f}], ||F(x)|| = {residual_norm:.2e}")
        return x, errors, iterations_count
    
    def newton_method(self, x0, epsilon=1e-6, max_iter=100):
        """
        Метод Ньютона
        x0 - начальное приближение
        epsilon - точность
        max_iter - максимальное число итераций
        """
        print("\nМетод Ньютона:")
        print(f"  Начальное приближение: {x0}")
        print(f"  Требуемая точность: {epsilon:.1e}")
        
        x = np.array(x0, dtype=float)
        errors = []
        iterations_count = []
        
        for i in range(max_iter):
            F = self.system(x)
            J = self.jacobian(x)
            
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError as e:
                print(f"  Ошибка: Матрица Якоби вырождена на итерации {i+1}. {e}")
                return None, errors, iterations_count
            
            x_new = x + dx
            
            error = np.linalg.norm(dx)
            errors.append(error)
            iterations_count.append(i + 1)
            
            print(f"  Итерация {i+1:<3}: x=[{x_new[0]:.8f}, {x_new[1]:.8f}], |dx|={error:.2e}")
            
            x = x_new
            
            if error < epsilon:
                residual_norm = np.linalg.norm(self.system(x))
                print(f"\n  Сходимость достигнута за {i+1} итераций.")
                print(f"  Решение: x = [{x[0]:.10f}, {x[1]:.10f}]")
                print(f"  Норма невязки ||F(x)|| = {residual_norm:.2e}")
                return x, errors, iterations_count
        
        print(f"\n  Метод не сошелся за {max_iter} итераций.")
        residual_norm = np.linalg.norm(self.system(x))
        print(f"  Последнее приближение: x = [{x[0]:.10f}, {x[1]:.10f}], ||F(x)|| = {residual_norm:.2e}")
        return x, errors, iterations_count

def plot_system_with_solution(a=3):
    """Графическое определение начального приближения с точным решением"""
    x1 = np.linspace(0, 4, 400)
    x2 = np.linspace(0, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    F1 = X1**2 + X2**2 - a**2
    F2 = X1 - np.exp(X2) + a
    
    from scipy.optimize import fsolve
    def equations(vars):
        x1, x2 = vars
        eq1 = x1**2 + x2**2 - a**2
        eq2 = x1 - np.exp(x2) + a
        return [eq1, eq2]
    
    initial_guesses = [[2.5, 1.5], [2.7, 1.3], [2.3, 1.7]]
    solutions = []
    
    for guess in initial_guesses:
        try:
            solution = fsolve(equations, guess)
            if solution[0] > 0 and solution[1] > 0:
                solutions.append(solution)
        except:
            continue
    
    unique_solutions = []
    for sol in solutions:
        if not any(np.allclose(sol, usol, atol=1e-3) for usol in unique_solutions):
            unique_solutions.append(sol)
    
    if unique_solutions:
        print(f"  Найденные fsolve решения: {[[f'{s[0]:.6f}', f'{s[1]:.6f}'] for s in unique_solutions]}")
    else:
        print("  fsolve не нашел положительных решений.")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contour(X1, X2, F1, levels=[0], colors='red', linewidths=2, label='x1² + x2² = 9')
    plt.contour(X1, X2, F2, levels=[0], colors='blue', linewidths=2, label='x1 - e^x2 + 3 = 0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Система уравнений')
    plt.axis('equal')
    
    for i, sol in enumerate(unique_solutions):
        plt.plot(sol[0], sol[1], 'go', markersize=10, label=f'Решение {i+1}' if i == 0 else "")
        plt.text(sol[0] + 0.1, sol[1] + 0.1, f'({sol[0]:.2f}, {sol[1]:.2f})', 
                fontsize=10, ha='left')
    
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if unique_solutions:
        sol = unique_solutions[0]
        x1_detailed = np.linspace(sol[0] - 0.5, sol[0] + 0.5, 400)
        x2_detailed = np.linspace(sol[1] - 0.5, sol[1] + 0.5, 400)
        X1_d, X2_d = np.meshgrid(x1_detailed, x2_detailed)
        
        F1_d = X1_d**2 + X2_d**2 - a**2
        F2_d = X1_d - np.exp(X2_d) + a
        
        plt.contour(X1_d, X2_d, F1_d, levels=[0], colors='red', linewidths=2, label='x1² + x2² = 9')
        plt.contour(X1_d, X2_d, F2_d, levels=[0], colors='blue', linewidths=2, label='x1 - e^x2 + 3 = 0')
        plt.plot(sol[0], sol[1], 'go', markersize=10)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Область решения (увеличенно)')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return unique_solutions[0] if unique_solutions else [2.5, 1.5]

if __name__ == "__main__":
    a = 3
    solver = NonlinearSystemSolver(a)
    
    print("Графическое определение начального приближения...")
    true_solution = plot_system_with_solution(a)
    
    print(f"\nНачальное приближение из графика: {true_solution}")
    
    print("\n" + "="*60)
    print("РЕШЕНИЕ СИСТЕМЫ УРАВНЕНИЙ")
    print("="*60)
    
    x0 = [2.7, 1.3]
    
    print("\n1. МЕТОД НЬЮТОНА:")
    x_newton, errors_newton, iters_newton = solver.newton_method(x0, 1e-10)
    
    print("\n2. МЕТОД ПРОСТОЙ ИТЕРАЦИИ:")
    x_simple, errors_simple, iters_simple = solver.simple_iteration_method(x0, 1e-6)
    
    print("\n" + "="*50)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("="*50)
    print(f"Метод Ньютона: {len(iters_newton)} итераций, невязка = {np.linalg.norm(solver.system(x_newton)):.2e}")
    print(f"Метод простой итерации: {len(iters_simple)} итераций, невязка = {np.linalg.norm(solver.system(x_simple)):.2e}")
    print(f"Разница между решениями: {np.linalg.norm(x_newton - x_simple):.2e}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(iters_simple, errors_simple, 'o-', label='Метод простой итерации', linewidth=2)
    plt.semilogy(iters_newton, errors_newton, 's-', label='Метод Ньютона', linewidth=2)
    plt.xlabel('Количество итераций')
    plt.ylabel('Погрешность')
    plt.title('Сравнение сходимости методов')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    max_iters = min(20, len(iters_newton), len(iters_simple))
    plt.semilogy(iters_simple[:max_iters], errors_simple[:max_iters], 'o-', 
                label='Метод простой итерации', linewidth=2)
    plt.semilogy(iters_newton[:max_iters], errors_newton[:max_iters], 's-', 
                label='Метод Ньютона', linewidth=2)
    plt.xlabel('Количество итераций')
    plt.ylabel('Погрешность')
    plt.title('Первые итерации (детально)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()