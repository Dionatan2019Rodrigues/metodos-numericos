def criterio_sassenfeld(A):
    n = len(A)
    betas = []
    for i in range(n):
        if A[i][i] == 0:
            return False, float('inf'), []
        soma_anteriores = sum(abs(A[i][j]) * betas[j] for j in range(i))
        soma_posteriores = sum(abs(A[i][j]) for j in range(i + 1, n))
        beta_i = (soma_anteriores + soma_posteriores) / abs(A[i][i])
        betas.append(beta_i)
    max_beta = max(betas)
    return max_beta < 1.0, max_beta, betas

def gauss_seidel(A, b, x0=None, tol=1e-5, max_iter=100):
    sol, _, _ = gauss_seidel_detalhado(A, b, x0, tol, max_iter)
    return sol

def gauss_seidel_detalhado(A, b, x0=None, tol=1e-5, max_iter=100):
    n = len(b)
    for i in range(n):
        if A[i][i] == 0:
            return None, [], False

    x = [float(val) for val in x0] if x0 else [0.0] * n
    historico = [{'iter': 0, 'x': x[:], 'erro': None}]

    convergiu = False
    for k in range(1, max_iter + 1):
        x_ant = x[:]
        for i in range(n):
            soma1 = sum(A[i][j] * x[j] for j in range(i))
            soma2 = sum(A[i][j] * x_ant[j] for j in range(i + 1, n))
            x[i] = (b[i] - soma1 - soma2) / A[i][i]

        denom = max(abs(val) for val in x)
        if denom == 0 or denom < 1e-12:
            erro = max(abs(x[i] - x_ant[i]) for i in range(n))
        else:
            erro = max(abs(x[i] - x_ant[i]) for i in range(n)) / denom

        historico.append({'iter': k, 'x': x[:], 'erro': erro})

        if erro < tol:
            convergiu = True
            return x, historico, convergiu

        if erro > 1e10 or any(abs(val) > 1e12 for val in x):
            return x, historico, False

    return x, historico, convergiu
