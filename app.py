import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laplace Lab", layout="centered")

st.title("Laplace Lab")
st.subheader("Resolução de EDOs usando Transformada de Laplace")

st.write("""
Este aplicativo resolve uma equação diferencial ordinária linear de segunda ordem
usando a Transformada de Laplace.
""")

st.latex(r"a y''(t) + b y'(t) + c y(t) = F(t)")

st.sidebar.header("Parâmetros da EDO")

a = st.sidebar.number_input("Coeficiente de y''", value=1.0)
b = st.sidebar.number_input("Coeficiente de y'", value=3.0)
c = st.sidebar.number_input("Coeficiente de y", value=2.0)

y0 = st.sidebar.number_input("Condição inicial y(0)", value=0.0)
v0 = st.sidebar.number_input("Condição inicial y'(0)", value=0.0)

tipo_forca = st.sidebar.selectbox(
    "Força externa F(t)",
    ["Constante", "Senoidal", "Exponencial"]
)

valor_forca = st.sidebar.number_input("Parâmetro da força", value=1.0)

t, s = sp.symbols("t s")
Y = sp.symbols("Y")

if tipo_forca == "Constante":
    F_t = valor_forca
elif tipo_forca == "Senoidal":
    F_t = sp.sin(valor_forca * t)
else:
    F_t = sp.exp(valor_forca * t)

st.subheader("Equação escolhida")

st.latex(
    rf"{a}y''(t)+{b}y'(t)+{c}y(t)={sp.latex(F_t)}"
)

st.latex(
    rf"y(0)={y0}, \quad y'(0)={v0}"
)

F_s = sp.laplace_transform(F_t, t, s, noconds=True)

equacao_laplace = sp.Eq(
    a * (s**2 * Y - s * y0 - v0)
    + b * (s * Y - y0)
    + c * Y,
    F_s
)

st.subheader("Equação no domínio de Laplace")

st.latex(sp.latex(equacao_laplace))

Y_s = sp.solve(equacao_laplace, Y)[0]
Y_s = sp.simplify(Y_s)

st.subheader("Solução no domínio de Laplace")

st.latex(r"Y(s)=" + sp.latex(Y_s))

y_t = sp.inverse_laplace_transform(Y_s, s, t)
y_t = sp.simplify(y_t)

st.subheader("Solução no domínio do tempo")

st.latex(r"y(t)=" + sp.latex(y_t))

st.subheader("Gráfico da solução")

try:
    y_func = sp.lambdify(t, y_t, "numpy")

    tempo = np.linspace(0, 10, 400)
    valores = y_func(tempo)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tempo, valores, linewidth=2)
    ax.set_xlabel("Tempo t")
    ax.set_ylabel("y(t)")
    ax.set_title("Solução da EDO")
    ax.grid(True)

    st.pyplot(fig)

except Exception as e:
    st.warning("Não foi possível gerar o gráfico para essa combinação de parâmetros.")
    st.write(e)

st.subheader("Interpretação")

st.write("""
A Transformada de Laplace transforma uma EDO, que envolve derivadas, em uma equação algébrica.
Depois de resolver essa equação para Y(s), aplicamos a transformada inversa para encontrar y(t).
""")
