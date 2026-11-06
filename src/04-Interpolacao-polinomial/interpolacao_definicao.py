def interpolacao_definicao(x_pts, y_pts, x_eval):
    resultado, _, _ = interpolacao_definicao_detalhada(x_pts, y_pts, x_eval)
    return resultado

def interpolacao_definicao_detalhada(x_pts, y_pts, x_eval=None):
    n = len(x_pts)
    if n != len(y_pts) or n == 0:
        return None, [], []

    V = [[float(x_pts[i] ** j) for j in range(n)] for i in range(n)]
    b = [float(y) for y in y_pts]

    A = [row[:] for row in V]
    rhs = b[:]

    for i in range(n):
        max_idx = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_idx][i]):
                max_idx = k
        A[i], A[max_idx] = A[max_idx], A[i]
        rhs[i], rhs[max_idx] = rhs[max_idx], rhs[i]

        if abs(A[i][i]) < 1e-14:
            return None, [], V

        for j in range(i + 1, n):
            fator = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= fator * A[i][k]
            rhs[j] -= fator * rhs[i]

    coefs = [0.0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(A[i][j] * coefs[j] for j in range(i + 1, n))
        coefs[i] = (rhs[i] - soma) / A[i][i]

    resultado = None
    if x_eval is not None:
        resultado = sum(coefs[j] * (x_eval ** j) for j in range(n))

    return resultado, coefs, V
