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

def bissecao(expr, a, b, tol, max_iter=100):
    expr = sanitiza_expressao(expr)
    fa = func(a, expr)
    fb = func(b, expr)

    if abs(fa) < 1e-15:
        return a, [{'iter': 1, 'a': a, 'b': b, 'x': a, 'y': fa, 'erro': 0.0}]
    if abs(fb) < 1e-15:
        return b, [{'iter': 1, 'a': a, 'b': b, 'x': b, 'y': fb, 'erro': 0.0}]

    if fa * fb > 0:
        return None, []

    historico = []

    for i in range(1, max_iter + 1):
        meio = (a + b) / 2.0
        f_meio = func(meio, expr)
        erro = abs(b - a) / 2.0

        historico.append({
            'iter': i,
            'a': a,
            'b': b,
            'x': meio,
            'y': f_meio,
            'erro': erro
        })

        if abs(f_meio) < tol or erro < tol:
            return meio, historico

        if fa * f_meio < 0:
            b = meio
            fb = f_meio
        else:
            a = meio
            fa = f_meio

    return (a + b) / 2.0, historico
