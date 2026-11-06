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

def secantes(expr, x0, x1, tol, max_iter=100):
    expr = sanitiza_expressao(expr)
    historico = []
    x_prev = float(x0)
    x_curr = float(x1)
    f_prev = func(x_prev, expr)
    f_curr = func(x_curr, expr)

    historico.append({
        'iter': 0,
        'x0': x_prev,
        'x1': x_prev,
        'fx0': f_prev,
        'fx1': f_prev,
        'x': x_prev,
        'y': f_prev,
        'erro': None
    })
    historico.append({
        'iter': 1,
        'x0': x_prev,
        'x1': x_curr,
        'fx0': f_prev,
        'fx1': f_curr,
        'x': x_curr,
        'y': f_curr,
        'erro': abs(x_curr - x_prev)
    })

    if abs(f_curr) < tol or abs(x_curr - x_prev) < tol:
        return x_curr, historico

    for i in range(2, max_iter + 1):
        den = f_curr - f_prev
        if abs(den) < 1e-15:
            return None, historico

        x_next = x_curr - f_curr * (x_curr - x_prev) / den
        f_next = func(x_next, expr)
        erro = abs(x_next - x_curr)

        historico.append({
            'iter': i,
            'x0': x_prev,
            'x1': x_curr,
            'fx0': f_prev,
            'fx1': f_curr,
            'x': x_next,
            'y': f_next,
            'erro': erro
        })

        if abs(f_next) < tol or erro < tol:
            return x_next, historico

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return x_curr, historico
