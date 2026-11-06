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

def regra_simpson_13(expr, a, b, n):
    area, _, _, _, _, _ = regra_simpson_detalhada(expr, a, b, n)
    return area

def regra_simpson_detalhada(expr, a, b, n):
    expr = sanitiza_expressao(expr)
    n = int(n)
    if n % 2 != 0:
        return None, None, [], 0.0, 0.0, []

    h = (b - a) / n
    x_nodes = [a + i * h for i in range(n + 1)]
    y_nodes = [func(xi, expr) for xi in x_nodes]

    tabela_passos = []
    soma_impares = 0.0
    soma_pares = 0.0

    for i in range(n + 1):
        if i == 0 or i == n:
            peso = 1
        elif i % 2 != 0:
            peso = 4
            soma_impares += y_nodes[i]
        else:
            peso = 2
            soma_pares += y_nodes[i]

        termo = peso * y_nodes[i]
        tabela_passos.append({
            'i': i,
            'xi': x_nodes[i],
            'f(xi)': y_nodes[i],
            'peso (c_i)': peso,
            'c_i * f(xi)': termo
        })

    soma_total = sum(row['c_i * f(xi)'] for row in tabela_passos)
    integral = (h / 3.0) * soma_total

    segmentos = []
    for k in range(0, n, 2):
        segmentos.append({
            'x_pts': [x_nodes[k], x_nodes[k + 1], x_nodes[k + 2]],
            'y_pts': [y_nodes[k], y_nodes[k + 1], y_nodes[k + 2]]
        })

    return integral, h, tabela_passos, soma_impares, soma_pares, segmentos
