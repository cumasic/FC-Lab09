import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del sistema de Lorenz
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Ecuaciones de Lorenz
def lorenz(estado_inicial , t):
	x, y, z = estado_inicial 
	dxdt = sigma * (y - x)
	dydt = x * (rho - z) - y
	dzdt = x * y - beta * z
	return [dxdt, dydt, dzdt]

# Condiciones iniciales
estado_inicial = [1.0, 1.0, 1.0]

# Intervalo de tiempo
t = np.linspace(0,100,10000)    

# Resolver las ecuaciones diferenciales
solucion = odeint(lorenz,estado_inicial ,t)

# Extraer soluciones
x = solucion[:,0]
y = solucion[:,1]
z = solucion[:,2]   

#Gráfica
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(x,y,z, lw=0.5)
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
ax.set_zlabel("Eje Z")
ax.set_title("Atractor de Lorenz")
plt.show()

# Parámetros de la gráfica
num_secciones = 15
z_secciones = np.linspace(11, 25, num_secciones)
tolerancia = 0.1

# Crear subplots
fig, axs = plt.subplots(3, 5, figsize=(20, 20))
fig.suptitle('Secciones de Poincaré del sistema de Lorenz', fontsize=20)

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


