import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del modelo
alpha = 0.1  # Tasa de crecimiento de las presas
beta = 0.02   # Tasa a la que los depredadores matan a las presas
gamma = 0.3  # Tasa de mortalidad de los depredadores
delta = 0.01  # Tasa a la que los depredadores aumentan por comer presas

# Definir las ecuaciones de Lotka-Volterra
def lotka_volterra(z, t):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Condiciones iniciales
x0 = 40 #Numero de presas
y0 = 9 #Numero de depredadores
z0 = [x0, y0]

# Tiempo de simulación
t = np.linspace(0, 200, 800)

# Resolver las ecuaciones diferenciales
solucion = odeint(lotka_volterra, z0, t)

# Extraer soluciones
x = solucion[:,0]
y = solucion[:,1]

# Sección de Poincaré (tomamos y = 9,10,11)
secciones_poincare = {y: [] for y in range(y0-1, y0+2)}

for y_seccion in secciones_poincare.keys():
    for i in range(1, len(t)):
        if solucion[i-1, 1] < y_seccion and solucion[i, 1] >= y_seccion:
            secciones_poincare[y_seccion].append(solucion[i, 0])

# Graficar los resultados
plt.figure(figsize=(16, 8))

# Trayectorias en el espacio de fases
plt.subplot(2, 2, 1)
x_max = np.max(solucion[:,0]) * 1.05
y_max = np.max(solucion[:,1]) * 1.05
x = np.linspace(0, x_max, 25)
y = np.linspace(0, y_max, 25)
xx, yy = np.meshgrid(x, y)
uu, vv = lotka_volterra((xx, yy),0)
norm = np.sqrt(uu**2 + vv**2)
uu = uu / norm
vv = vv / norm
plt.quiver(xx, yy, uu, vv, norm, cmap=plt.cm.gray)
plt.plot(solucion[:, 0], solucion[:, 1])
plt.xlabel('Presas (x)')
plt.ylabel('Depredadores (y)')
plt.title('Espacio de fases')

# Sección de Poincaré
plt.subplot(2, 2, 2)
plt.scatter(secciones_poincare[y0-1], np.zeros_like(secciones_poincare[y0-1]))
plt.xlabel('Presas (x)')
plt.title(f'Sección de Poincaré en y={y0-1}')
plt.grid()

plt.subplot(2, 2, 3)
plt.scatter(secciones_poincare[y0], np.zeros_like(secciones_poincare[y0]))
plt.xlabel('Presas (x)')
plt.title(f'Sección de Poincaré en y={y0}')
plt.grid()

plt.subplot(2, 2, 4)
plt.scatter(secciones_poincare[y0+1], np.zeros_like(secciones_poincare[y0+1]))
plt.xlabel('Presas (x)')
plt.title(f'Sección de Poincaré en y={y0+1}')
plt.grid()

plt.tight_layout()
plt.show()
