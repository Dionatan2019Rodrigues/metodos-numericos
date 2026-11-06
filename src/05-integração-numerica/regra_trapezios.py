import math
from sanitizador import sanitiza_expressao

def avalia_funcao_segura(expr, x_val):
    namespace = {
        "x": float(x_val),
        "math": math,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs
    }
    return float(eval(expr, {"__builtins__": {}}, namespace))

def func(x, expr):
    return avalia_funcao_segura(expr, x)

def regra_trapezios_repetida(expr, a, b, n):
    area, _, _, _, _ = regra_trapezios_detalhada(expr, a, b, n)
    return area

def regra_trapezios_detalhada(expr, a, b, n):
    expr = sanitiza_expressao(expr)
    n = int(n)
    h = (b - a) / n
    x_nodes = [a + i * h for i in range(n + 1)]
    y_nodes = [func(xi, expr) for xi in x_nodes]

    tabela_passos = []
    soma_interior = 0.0

    for i in range(n + 1):
        peso = 1 if (i == 0 or i == n) else 2
        termo = peso * y_nodes[i]
        if i not in (0, n):
            soma_interior += y_nodes[i]

        tabela_passos.append({
            'i': i,
            'xi': x_nodes[i],
            'f(xi)': y_nodes[i],
            'peso (c_i)': peso,
            'c_i * f(xi)': termo
        })

    soma_total = sum(row['c_i * f(xi)'] for row in tabela_passos)
    integral = (h / 2.0) * soma_total

    subintervalos = []
    for i in range(n):
        subintervalos.append({
            'x0': x_nodes[i],
            'x1': x_nodes[i + 1],
            'y0': y_nodes[i],
            'y1': y_nodes[i + 1],
            'area_trap': (h / 2.0) * (y_nodes[i] + y_nodes[i + 1])
        })

    return integral, h, tabela_passos, soma_interior, subintervalos
