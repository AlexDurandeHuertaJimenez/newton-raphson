#Para COMPILAR y EJECUTAR el programa copia y pega el siguiente comando:
# python newton_raphson.py

#En caso de no tener instaladas las bibliotecas necesarias utiliza este comando:
# pip install sympy numpy

#Se importan las bibliotecas necesarias
import sympy as sp  # Para el cálculo simbólico y derivadas
import numpy as np   # Para manejar operaciones con arrays y matrices numéricas

# Función principal que resuelve el sistema de ecuaciones utilizando el método de Newton-Raphson
def newton_raphson_system(equations, initial_guess, tolerance=1e-7, max_iterations=100):
    # Definir las variables simbólicas x e y
    x, y = sp.symbols('x y')
    
    # Las ecuaciones son pasadas como una tupla, así que las desempaquetamos
    f1, f2 = equations

    # Crear la matriz Jacobiana, que contiene las derivadas parciales de las ecuaciones
    J = sp.Matrix([[sp.diff(f1, x), sp.diff(f1, y)],  # Derivadas parciales de la primera ecuación
                   [sp.diff(f2, x), sp.diff(f2, y)]])  # Derivadas parciales de la segunda ecuación
    
    # Convertir las ecuaciones simbólicas y la Jacobiana a funciones numéricas
    f1_func = sp.lambdify((x, y), f1, 'numpy')  # Función que evalúa f1 numéricamente
    f2_func = sp.lambdify((x, y), f2, 'numpy')  # Función que evalúa f2 numéricamente
    J_func = sp.lambdify((x, y), J, 'numpy')    # Función que evalúa la Jacobiana numéricamente

    # Convertir la conjetura inicial en un array de tipo float para cálculos numéricos
    guess = np.array(initial_guess, dtype=float)
    
    # Lista para almacenar los resultados de cada iteración
    iterations = []

    # Bucle que realiza el proceso iterativo de Newton-Raphson
    for i in range(max_iterations):
        try:
            # Evaluar las funciones y la Jacobiana en el punto actual (guess)
            f_val = np.array([f1_func(*guess), f2_func(*guess)], dtype=float)  # Valor de las ecuaciones
            J_val = np.array(J_func(*guess), dtype=float)  # Valor de la Jacobiana
        except Exception as e:
            print(f"Error evaluando funciones en la iteración {i + 1}: {e}")
            return None, iterations  # En caso de error, retorna None y las iteraciones ya realizadas
        
        try:
            # Resolver el sistema lineal J * delta = -f, para obtener el cambio en la solución
            delta = np.linalg.solve(J_val, -f_val)  # Resolvemos para obtener el cambio delta
        except np.linalg.LinAlgError:
            print("La matriz Jacobiana es singular. No se puede continuar.")
            return None, iterations  # Si no se puede resolver el sistema, retornamos None

        # Actualizamos el valor de la solución sumando el cambio (delta) al valor actual
        next_guess = guess + delta
        # Almacenamos la información de la iteración
        iterations.append((i + 1, guess[0], guess[1], f_val[0], f_val[1]))

        # Verificar si el cambio es suficientemente pequeño (criterio de convergencia)
        if np.all(np.abs(delta) < tolerance):
            return next_guess, iterations  # Si convergemos, retornamos la solución y las iteraciones
        
        # Si no hemos convergido, actualizamos el valor de la conjetura y seguimos con la siguiente iteración
        guess = next_guess

    # Si no encontramos la solución en el número máximo de iteraciones, lo indicamos
    print("No se encontró una solución dentro del número máximo de iteraciones.")
    return None, iterations

# Función principal donde el usuario ingresa las ecuaciones y las conjeturas iniciales
def main():
    # Definir las variables simbólicas x e y
    x, y = sp.symbols('x y')

    try:
        # Pedir al usuario las ecuaciones
        eq1 = input("Ingrese la primera ecuación (en términos de x e y): ").replace('^', '**')  # Reemplazar ^ por **
        eq2 = input("Ingrese la segunda ecuación (en términos de x e y): ").replace('^', '**')  # Reemplazar ^ por **
        
        # Convertir las ecuaciones de texto a expresiones simbólicas
        eq1 = sp.sympify(eq1) - 0  # Aseguramos que las ecuaciones estén igualadas a cero
        eq2 = sp.sympify(eq2) - 0  # Aseguramos que las ecuaciones estén igualadas a cero
    except Exception as e:
        print(f"Error al interpretar las ecuaciones: {e}")
        return  # Si ocurre un error, terminamos la ejecución

    try:
        # Pedir las conjeturas iniciales (valores de x e y)
        initial_guess_x = float(input("Ingrese la conjetura inicial para x: "))
        initial_guess_y = float(input("Ingrese la conjetura inicial para y: "))
    except ValueError:
        print("Error: Las conjeturas iniciales deben ser números.")
        return  # Si hay un error con las conjeturas, terminamos la ejecución

    # Pedir al usuario el número de decimales a mostrar
    decimals = int(input("Ingrese la cantidad de decimales para los resultados: "))

    # Llamar a la función de Newton-Raphson para resolver el sistema de ecuaciones
    root, iterations = newton_raphson_system((eq1, eq2), (initial_guess_x, initial_guess_y))

    if root is not None:
        # Si encontramos la solución, mostramos el resultado
        print(f"\nSolución encontrada: x = {root[0]:.{decimals}f}, y = {root[1]:.{decimals}f}")
        print("Iteraciones:")
        # Mostramos las iteraciones y los resultados de cada una
        print("Iteración | x actual | y actual | f1(x, y)       | f2(x, y)")
        for iteration in iterations:
            iter_num = iteration[0]  # Número de la iteración
            x_val = iteration[1]     # Valor de x en la iteración
            y_val = iteration[2]     # Valor de y en la iteración
            f1_val = iteration[3]    # Valor de la primera ecuación en esa iteración
            f2_val = iteration[4]    # Valor de la segunda ecuación en esa iteración
            # Usamos formato para eliminar los ceros innecesarios después del decimal
            print(f"{iter_num:<10} | {x_val:.{decimals}f} | {y_val:.{decimals}f} | {f1_val:.{decimals}f} | {f2_val:.{decimals}f}")
    else:
        print("No se encontró solución para el sistema.")

# Ejecutar la función principal cuando se ejecuta el script
if __name__ == "__main__":
    main()
