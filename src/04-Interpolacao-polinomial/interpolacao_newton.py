import sympy as sp

def tabela_diferencas_divididas(x_pts, y_pts):
    n = len(y_pts)
    tabela = [[0.0] * n for _ in range(n)]
    for i in range(n):
        tabela[i][0] = y_pts[i]

    for j in range(1, n):
        for i in range(n - j):
            tabela[i][j] = (tabela[i + 1][j - 1] - tabela[i][j - 1]) / (x_pts[i + j] - x_pts[i])

    return tabela[0]

def tabela_completa_diferencas(x_pts, y_pts):
    n = len(y_pts)
    tabela = [[None] * n for _ in range(n)]
    for i in range(n):
        tabela[i][0] = float(y_pts[i])

    for j in range(1, n):
        for i in range(n - j):
            tabela[i][j] = (tabela[i + 1][j - 1] - tabela[i][j - 1]) / (x_pts[i + j] - x_pts[i])

    return tabela

def interpolacao_newton(x_pts, y_pts, x_eval):
    coefs = tabela_diferencas_divididas(x_pts, y_pts)
    n = len(coefs)
    resultado = coefs[0]
    produto = 1.0

    for i in range(1, n):
        produto *= (x_eval - x_pts[i - 1])
        resultado += coefs[i] * produto

    return resultado

def interpolacao_newton_detalhada(x_pts, y_pts, x_eval=None):
    tabela = tabela_completa_diferencas(x_pts, y_pts)
    coefs = [tabela[0][j] for j in range(len(x_pts))]

    x = sp.Symbol('x')
    P_sym = coefs[0]
    prod = 1
    for i in range(1, len(coefs)):
        prod = prod * (x - x_pts[i - 1])
        P_sym = P_sym + coefs[i] * prod

    P_simplificado = sp.expand(P_sym)
    resultado = float(P_simplificado.subs(x, x_eval)) if x_eval is not None else None

    return resultado, tabela, coefs, P_simplificado
