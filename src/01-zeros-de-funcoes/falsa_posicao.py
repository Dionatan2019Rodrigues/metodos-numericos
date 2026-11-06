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

def falsa_posicao(expr, a, b, tol, max_iter=100):
    expr = sanitiza_expressao(expr)
    fa = func(a, expr)
    fb = func(b, expr)

    if abs(fa) < 1e-15:
        return a, [{'iter': 1, 'a': a, 'b': b, 'fa': fa, 'fb': fb, 'x': a, 'y': fa, 'erro': 0.0}]
    if abs(fb) < 1e-15:
        return b, [{'iter': 1, 'a': a, 'b': b, 'fa': fa, 'fb': fb, 'x': b, 'y': fb, 'erro': 0.0}]

    if fa * fb > 0:
        return None, []

    historico = []
    ponto_ant = None

    for i in range(1, max_iter + 1):
        den = fb - fa
        if abs(den) < 1e-15:
            return None, historico

        ponto = (a * fb - b * fa) / den
        fp = func(ponto, expr)
        erro = abs(ponto - ponto_ant) if ponto_ant is not None else abs(b - a)

        historico.append({
            'iter': i,
            'a': a,
            'b': b,
            'fa': fa,
            'fb': fb,
            'x': ponto,
            'y': fp,
            'erro': erro
        })

        if abs(fp) < tol or (ponto_ant is not None and erro < tol):
            return ponto, historico

        if fa * fp < 0:
            b = ponto
            fb = fp
        else:
            a = ponto
            fa = fp

        ponto_ant = ponto

    return ponto, historico
