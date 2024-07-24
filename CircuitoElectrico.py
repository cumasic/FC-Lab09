import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parámetros del circuito
alpha = 15.6
beta = 28.0
m0 = -1.143
m1 = -0.714

# Definimos la función no lineal f(x)
def f(x, m0, m1):
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

# Definimos las ecuaciones diferenciales del circuito de Chua
def circuito_chua(estado_inicial, t):
    x, y, z = estado_inicial
    dxdt = alpha * (y - x - f(x, m0, m1))
    dydt = x - y + z
    dzdt = -beta * y
    return [dxdt, dydt, dzdt]

# Condiciones iniciales
estado_inicial = [0.7, 0.0, 0.0]

# Intervalo de tiempo
t = np.linspace(0, 100, 10000)

# Resolver las ecuaciones diferenciales
solucion = odeint(circuito_chua, estado_inicial, t)

# Extraer soluciones
x = solucion[:,0]
y = solucion[:,1]
z = solucion[:,2]  

# Graficar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Atractor de Chua')
plt.show()

# Parámetros de la gráfica
num_secciones = 15
z_secciones = np.linspace(0, 3, num_secciones)
tolerancia = 0.1

# Crear subplots
fig, axs = plt.subplots(3, 5, figsize=(20, 20))
fig.suptitle('Secciones de Poincaré del circuito de Chua', fontsize=20)

# Generar y graficar secciones de Poincaré para diferentes planos z
for idx, z_seccion in enumerate(z_secciones):
    poincare_puntos_x = []
    poincare_puntos_y = []
    for i in range(1, len(z)):
        if (z[i-1] < z_seccion and z[i] >= z_seccion) or (z[i-1] > z_seccion and z[i] <= z_seccion):
            if abs(z[i] - z_seccion) < tolerancia:
                poincare_puntos_x.append(x[i])
                poincare_puntos_y.append(y[i])

    ax = axs[idx // 5, idx % 5]
    ax.plot(poincare_puntos_x, poincare_puntos_y, 'o', markersize=2)
    ax.set_title(f'z = {z_seccion:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()

plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()