import math
import sympy as sp
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

def derivada(x_val, expr, h=1e-7):
    try:
        x_sym = sp.Symbol('x')
        expr_sym = sp.sympify(expr)
        df_sym = sp.diff(expr_sym, x_sym)
        return float(df_sym.subs(x_sym, x_val))
    except Exception:
        return (func(x_val + h, expr) - func(x_val - h, expr)) / (2.0 * h)

def newton(expr, chute, tol, max_iter=100):
    expr = sanitiza_expressao(expr)
    historico = []
    x_atual = float(chute)

    for i in range(1, max_iter + 1):
        y_atual = func(x_atual, expr)
        df = derivada(x_atual, expr)

        if abs(df) < 1e-15:
            historico.append({
                'iter': i,
                'x': x_atual,
                'y': y_atual,
                'df': df,
                'x_prox': None,
                'erro': None
            })
            return None, historico

        x_proximo = x_atual - y_atual / df
        erro = abs(x_proximo - x_atual)

        historico.append({
            'iter': i,
            'x': x_atual,
            'y': y_atual,
            'df': df,
            'x_prox': x_proximo,
            'erro': erro
        })

        if abs(y_atual) < tol or erro < tol:
            return x_proximo, historico

        x_atual = x_proximo

    return x_atual, historico
