import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

def sanitiza_expressao(s):
    if not isinstance(s, str):
        return str(s)
    s = s.strip()
    s = s.replace('ˆ', '^').replace('˜', '')
    s = re.sub(r'(\d+),(\d+)', r'\1.\2', s)
    s = re.sub(r'\bsen\b', 'sin', s, flags=re.IGNORECASE)
    s = re.sub(r'\btg\b', 'tan', s, flags=re.IGNORECASE)
    s = re.sub(r'\bln\b', 'log', s, flags=re.IGNORECASE)
    s = re.sub(r'\braiz\b', 'sqrt', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\^\((.*?)\)', r'exp(\1)', s)
    s = re.sub(r'\be\^([a-zA-Z0-9_\-]+)', r'exp(\1)', s)

    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    try:
        expr_sym = parse_expr(s, transformations=transformations)
        return str(expr_sym)
    except Exception:
        return s.replace('^', '**')

def processa_expressao_natural(texto):
    if not texto or not texto.strip():
        return None, '', 'Entrada vazia.'
    s = texto.strip()
    s = s.replace('ˆ', '^').replace('˜', '')
    s = re.sub(r'(\d+),(\d+)', r'\1.\2', s)
    s = re.sub(r'\bsen\b', 'sin', s, flags=re.IGNORECASE)
    s = re.sub(r'\btg\b', 'tan', s, flags=re.IGNORECASE)
    s = re.sub(r'\bln\b', 'log', s, flags=re.IGNORECASE)
    s = re.sub(r'\braiz\b', 'sqrt', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\^\((.*?)\)', r'exp(\1)', s)
    s = re.sub(r'\be\^([a-zA-Z0-9_\-]+)', r'exp(\1)', s)

    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    try:
        expr_sym = parse_expr(s, transformations=transformations)
        expr_py = str(expr_sym)
        expr_latex = sp.latex(expr_sym)
        return expr_py, expr_latex, None
    except Exception as e:
        expr_fallback = s.replace('^', '**')
        try:
            expr_sym = sp.sympify(expr_fallback)
            return str(expr_sym), sp.latex(expr_sym), None
        except Exception:
            return None, '', str(e)

