import sympy as sp

def interpolacao_lagrange(x_pts, y_pts, x_eval):
    n = len(x_pts)
    resultado = 0.0

    for i in range(n):
        termo = y_pts[i]
        for j in range(n):
            if i != j:
                termo *= (x_eval - x_pts[j]) / (x_pts[i] - x_pts[j])
        resultado += termo

    return resultado

def interpolacao_lagrange_detalhada(x_pts, y_pts, x_eval=None):
    n = len(x_pts)
    x = sp.Symbol('x')
    polinomios_L = []
    P_sym = 0

    for i in range(n):
        L_num = 1
        L_den = 1
        for j in range(n):
            if i != j:
                L_num = L_num * (x - x_pts[j])
                L_den = L_den * (x_pts[i] - x_pts[j])
        L_i = L_num / L_den
        termo_i = y_pts[i] * L_i
        P_sym = P_sym + termo_i

        val_Li = None
        if x_eval is not None:
            val_Li = float(L_i.subs(x, x_eval))

        polinomios_L.append({
            'i': i,
            'xi': x_pts[i],
            'yi': y_pts[i],
            'L_expr': sp.simplify(L_i),
            'L_val': val_Li,
            'termo_val': val_Li * y_pts[i] if val_Li is not None else None
        })

    P_simplificado = sp.expand(P_sym)
    resultado = float(P_simplificado.subs(x, x_eval)) if x_eval is not None else None

    return resultado, polinomios_L, P_simplificado
