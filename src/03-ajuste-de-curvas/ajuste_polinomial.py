def ajuste_polinomial(x, y, grau):
    coefs, _, _, _, _, _, _ = ajuste_polinomial_detalhado(x, y, grau)
    return coefs

def ajuste_polinomial_detalhado(x, y, grau):
    n = len(x)
    m = grau + 1

    A = [[0.0] * m for _ in range(m)]
    B = [0.0] * m

    for i in range(m):
        for j in range(m):
            A[i][j] = sum(float(x[k])**(i + j) for k in range(n))
        B[i] = sum((float(x[k])**i) * float(y[k]) for k in range(n))

    matriz_original = [row[:] for row in A]
    vetor_original = B[:]

    A_elim = [row[:] for row in A]
    B_elim = B[:]

    for i in range(m):
        max_row = i
        for k in range(i + 1, m):
            if abs(A_elim[k][i]) > abs(A_elim[max_row][i]):
                max_row = k
        A_elim[i], A_elim[max_row] = A_elim[max_row], A_elim[i]
        B_elim[i], B_elim[max_row] = B_elim[max_row], B_elim[i]

        if abs(A_elim[i][i]) < 1e-14:
            return None, matriz_original, vetor_original, 0, 0, 0, []

        for j in range(i + 1, m):
            fator = A_elim[j][i] / A_elim[i][i]
            for k in range(i, m):
                A_elim[j][k] -= fator * A_elim[i][k]
            B_elim[j] -= fator * B_elim[i]

    c = [0.0] * m
    for i in range(m - 1, -1, -1):
        soma = sum(A_elim[i][j] * c[j] for j in range(i + 1, m))
        c[i] = (B_elim[i] - soma) / A_elim[i][i]

    coefs = c[::-1]

    y_medio = sum(y) / n
    sqt = sum((y[k] - y_medio)**2 for k in range(n))
    sqe = 0.0

    tabela_residuos = []
    for k in range(n):
        y_ajustado = sum(c[j] * (float(x[k])**j) for j in range(m))
        residuo = float(y[k]) - y_ajustado
        res_quadrado = residuo**2
        sqe += res_quadrado
        tabela_residuos.append({
            'x': float(x[k]),
            'y_real': float(y[k]),
            'y_ajustado': y_ajustado,
            'residuo': residuo,
            'residuo_quadrado': res_quadrado
        })

    r2 = 1.0 - (sqe / sqt) if sqt > 1e-12 else 1.0

    return coefs, matriz_original, vetor_original, r2, sqe, sqt, tabela_residuos
