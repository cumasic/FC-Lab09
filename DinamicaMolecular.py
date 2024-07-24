import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema
num_particulas = 3
tmn_caja = 10.0
epsilon = 1.0
sigma = 1.0
masa = 1.0
dt = 0.01
num_pasos = 10000
z_seccion = 5.0  # Plano z en el cual tomamos la sección de Poincaré

# Inicialización de posiciones y velocidades
np.random.seed(42)
posiciones = np.random.rand(num_particulas, 3) * tmn_caja
velocidades = np.random.randn(num_particulas, 3)

def calcular_fuerzas(posiciones):
    fuerzas = np.zeros_like(posiciones)
    for i in range(num_particulas):
        for j in range(i + 1, num_particulas):
            rij = posiciones[i] - posiciones[j]
            rij = rij - tmn_caja * np.round(rij / tmn_caja)  # PBC
            r2 = np.sum(rij**2)
            if r2 < 9.0:  # Cutoff radius
                r2_inv = 1.0 / r2
                r6_inv = r2_inv**3
                f = 24 * epsilon * r6_inv * (2 * r6_inv - 1) * r2_inv * rij
                fuerzas[i] += f
                fuerzas[j] -= f
    return fuerzas

def integrar(posiciones, velocidades, fuerzas, dt):
    velocidades += 0.5 * fuerzas * dt / masa
    posiciones += velocidades * dt
    posiciones = posiciones % tmn_caja  # PBC
    fuerzas = calcular_fuerzas(posiciones)
    velocidades += 0.5 * fuerzas * dt / masa
    return posiciones, velocidades, fuerzas

# Inicializar fuerzas
fuerzas = calcular_fuerzas(posiciones)

# Listas para almacenar la trayectoria y la sección de Poincaré
trayectorias = [[] for _ in range(num_particulas)]
secciones_poincare = [[] for _ in range(num_particulas)]

# Integración del sistema
for paso in range(num_pasos):
    posiciones, velocidades, fuerzas = integrar(posiciones, velocidades, fuerzas, dt)
    if paso % 100 == 0:
        for i in range(num_particulas):
            trayectorias[i].append(posiciones[i])  # Almacenar posiciones completas en 3D
            # Almacenar posición en la sección de Poincaré si z está cerca de z_section
            if abs(posiciones[i, 2] - z_seccion) < 0.1:
                secciones_poincare[i].append(posiciones[i, :2])  # Solo x e y

secciones_poincare = [np.array(section) for section in secciones_poincare]
trayectorias = [np.array(trajectory) for trajectory in trayectorias]

# Graficar las secciones de Poincaré
fig = plt.figure(figsize=(12, 8))

# Graficar las trayectorias de las partículas en 3D
ax2 = fig.add_subplot(1, 2, 1, projection='3d')
for i in range(num_particulas):
    traj = trayectorias[i]
    ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Partícula {i+1}')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Trayectorias de las Partículas')
ax2.legend()
ax2.grid(True)


ax = fig.add_subplot(1,2,2)
ax.scatter(secciones_poincare[0][:, 0], secciones_poincare[0][:, 1], s=1, color='blue')
ax.scatter(secciones_poincare[1][:, 0], secciones_poincare[1][:, 1], s=1, color='orange')
ax.scatter(secciones_poincare[2][:, 0], secciones_poincare[2][:, 1], s=1, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Sección de Poincaré de la Partícula {i+1} (z = {z_seccion})')
ax.grid(True)

plt.tight_layout()
plt.show()