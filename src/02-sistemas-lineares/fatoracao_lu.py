def fatoracao_lu(A, b):
    x, _, _, _ = fatoracao_lu_detalhada(A, b)
    return x

def fatoracao_lu_detalhada(A, b):
    n = len(b)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[float(val) for val in row] for row in A]

    for i in range(n):
        if abs(U[i][i]) < 1e-14:
            return None, L, U, None

        for j in range(i + 1, n):
            fator = U[j][i] / U[i][i]
            L[j][i] = fator
            for k in range(i, n):
                U[j][k] -= fator * U[i][k]

    y = [0.0] * n
    for i in range(n):
        soma = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - soma

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-14:
            return None, L, U, y
        soma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - soma) / U[i][i]

    return x, L, U, y
