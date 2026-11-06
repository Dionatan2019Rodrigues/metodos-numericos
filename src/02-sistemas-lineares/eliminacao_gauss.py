def eliminacao_gauss(A, b):
    x, _ = eliminacao_gauss_detalhada(A, b)
    return x

def eliminacao_gauss_detalhada(A, b):
    n = len(b)
    A_copia = [[float(val) for val in row] for row in A]
    b_copia = [float(val) for val in b]

    etapas = []
    matriz_inicial = [A_copia[i][:] + [b_copia[i]] for i in range(n)]
    etapas.append({
        'passo': 0,
        'descricao': 'Matriz Aumentada Inicial [A | b]',
        'matriz': matriz_inicial
    })

    for i in range(n):
        max_idx = i
        for k in range(i + 1, n):
            if abs(A_copia[k][i]) > abs(A_copia[max_idx][i]):
                max_idx = k

        troca = False
        if max_idx != i:
            A_copia[i], A_copia[max_idx] = A_copia[max_idx], A_copia[i]
            b_copia[i], b_copia[max_idx] = b_copia[max_idx], b_copia[i]
            troca = True

        if abs(A_copia[i][i]) < 1e-14:
            return None, etapas

        multiplicadores = {}
        for j in range(i + 1, n):
            fator = A_copia[j][i] / A_copia[i][i]
            multiplicadores[f'm_{j+1}{i+1}'] = fator
            for k in range(i, n):
                A_copia[j][k] -= fator * A_copia[i][k]
            b_copia[j] -= fator * b_copia[i]

        matriz_atual = [A_copia[r][:] + [b_copia[r]] for r in range(n)]
        desc = f"Coluna {i+1} escalonada (Pivô A[{i+1},{i+1}] = {A_copia[i][i]:.4f})"
        if troca:
            desc = f"Troca L{i+1} <-> L{max_idx+1} e " + desc

        etapas.append({
            'passo': i + 1,
            'descricao': desc,
            'multiplicadores': multiplicadores,
            'matriz': matriz_atual
        })

    x = [0.0] * n
    passos_subst = []
    for i in range(n - 1, -1, -1):
        soma = sum(A_copia[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b_copia[i] - soma) / A_copia[i][i]
        passos_subst.append({
            'variavel': f'x_{i+1}',
            'valor': x[i]
        })

    return x, etapas
