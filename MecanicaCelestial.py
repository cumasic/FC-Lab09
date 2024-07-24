import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

def problema_tres_cuerpos(t, estado_inicial, m1, m2):
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, x3, y3, z3, vx3, vy3, vz3 = estado_inicial
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
    
    ax1 = m2 * (x2 - x1) / r12**3
    ay1 = m2 * (y2 - y1) / r12**3
    az1 = m2 * (z2 - z1) / r12**3
    ax2 = m1 * (x1 - x2) / r12**3
    ay2 = m1 * (y1 - y2) / r12**3
    az2 = m1 * (z1 - z2) / r12**3
    ax3 = m1 * (x1 - x3) / r13**3 + m2 * (x2 - x3) / r23**3
    ay3 = m1 * (y1 - y3) / r13**3 + m2 * (y2 - y3) / r23**3
    az3 = m1 * (z1 - z3) / r13**3 + m2 * (z2 - z3) / r23**3

    return [vx1, vy1, vz1, ax1, ay1, az1, vx2, vy2, vz2, ax2, ay2, az2, vx3, vy3, vz3, ax3, ay3, az3]

def simulacion_problema_tres_cuerpos(m1, m2, estado_inicial, t_inter, t_eval):
    resultado = solve_ivp(problema_tres_cuerpos, t_inter, estado_inicial, args=(m1, m2), t_eval=t_eval, rtol=1e-9, atol=1e-9)
    return resultado

# Parámetros del problema
m1 = 1.0
m2 = 1.0

# Ajustar las posiciones y velocidades iniciales para incluir movimiento en z
estado_inicial = [
    0, 0, 0.1, 0, 0, 0.1,     # Cuerpo 1
    1, 0, -0.1, 0, 1, -0.1,   # Cuerpo 2
    0.5, 0.5, 0.1, 0, 0.5, 0.2 # Cuerpo 3
]

# Intervalo de tiempo y evaluación de intervalo
t_inter = (0, 20)
t_eval = np.linspace(0, 20, 1000)

# Simulación
resultado = simulacion_problema_tres_cuerpos(m1, m2, estado_inicial, t_inter, t_eval)

# Extraer datos para graficar
x1, y1, z1 = resultado.y[0], resultado.y[1], resultado.y[2]
x2, y2, z2 = resultado.y[6], resultado.y[7], resultado.y[8]
x3, y3, z3 = resultado.y[12], resultado.y[13], resultado.y[14]

# Graficar en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, y1, z1, label='Cuerpo 1')
ax.plot(x2, y2, z2, label='Cuerpo 2')
ax.plot(x3, y3, z3, label='Cuerpo 3')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Problema de los Tres Cuerpos Restringido en 3D')
ax.legend()
plt.show()


# Parámetros de la gráfica
num_secciones = 15
y_secciones = np.linspace(0, 5, num_secciones)
tolerancia = 0.1

# Crear subplots
fig, axs = plt.subplots(3, 5, figsize=(20, 20))
fig.suptitle('Secciones de Poincaré del problema de los 3 cuerpos restringido', fontsize=20)

# Generar y graficar secciones de Poincaré para diferentes planos y 
for idx, y_seccion in enumerate(y_secciones):
    poincare_puntos_c1_x = []
    poincare_puntos_c1_z = []
    for i in range(1, len(y1)):
        if (y1[i-1] < y_seccion and y1[i] >= y_seccion) or (y1[i-1] > y_seccion and y1[i] <= y_seccion):
            if abs(y1[i] - y_seccion) < tolerancia:
                poincare_puntos_c1_x.append(x1[i])
                poincare_puntos_c1_z.append(z1[i])

for idx, y_seccion in enumerate(y_secciones):
    poincare_puntos_c2_x = []
    poincare_puntos_c2_z = []
    for i in range(1, len(y2)):
        if (y2[i-1] < y_seccion and y2[i] >= y_seccion) or (y2[i-1] > y_seccion and y2[i] <= y_seccion):
            if abs(y2[i] - y_seccion) < tolerancia:
                poincare_puntos_c2_x.append(x2[i])
                poincare_puntos_c2_z.append(z2[i])

for idx, y_seccion in enumerate(y_secciones):
    poincare_puntos_c3_x = []
    poincare_puntos_c3_z = []
    for i in range(1, len(y3)):
        if (y3[i-1] < y_seccion and y3[i] >= y_seccion) or (y3[i-1] > y_seccion and y3[i] <= y_seccion):
            if abs(y3[i] - y_seccion) < tolerancia:
                poincare_puntos_c3_x.append(x3[i])
                poincare_puntos_c3_z.append(z3[i])

    # Graficar en el subplot correspondiente
    ax = axs[idx // 5, idx % 5]
    ax.plot(poincare_puntos_c1_x, poincare_puntos_c1_z, 'o', markersize=2)
    ax.plot(poincare_puntos_c2_x, poincare_puntos_c2_z, 'o', markersize=2)
    ax.plot(poincare_puntos_c3_x, poincare_puntos_c3_z, 'o', markersize=2)
    ax.set_title(f'y = {y_seccion:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.grid()

plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()
