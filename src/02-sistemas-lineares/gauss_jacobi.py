def criterio_linhas(A):
    n = len(A)
    alphas = []
    for i in range(n):
        if A[i][i] == 0:
            return False, float('inf'), []
        soma = sum(abs(A[i][j]) for j in range(n) if j != i)
        alpha_i = soma / abs(A[i][i])
        alphas.append(alpha_i)
    max_alpha = max(alphas)
    return max_alpha < 1.0, max_alpha, alphas

def gauss_jacobi(A, b, x0=None, tol=1e-5, max_iter=100):
    sol, _, _ = gauss_jacobi_detalhado(A, b, x0, tol, max_iter)
    return sol

def gauss_jacobi_detalhado(A, b, x0=None, tol=1e-5, max_iter=100):
    n = len(b)
    for i in range(n):
        if A[i][i] == 0:
            return None, [], False

    x = [float(val) for val in x0] if x0 else [0.0] * n
    x_novo = [0.0] * n
    historico = [{'iter': 0, 'x': x[:], 'erro': None}]

    convergiu = False
    for k in range(1, max_iter + 1):
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_novo[i] = (b[i] - soma) / A[i][i]

        denom = max(abs(val) for val in x_novo)
        if denom == 0 or denom < 1e-12:
            erro = max(abs(x_novo[i] - x[i]) for i in range(n))
        else:
            erro = max(abs(x_novo[i] - x[i]) for i in range(n)) / denom

        historico.append({'iter': k, 'x': x_novo[:], 'erro': erro})

        if erro < tol:
            convergiu = True
            return x_novo, historico, convergiu

        if erro > 1e10 or any(abs(val) > 1e12 for val in x_novo):
            return x_novo, historico, False

        x = x_novo[:]

    return x, historico, convergiu
