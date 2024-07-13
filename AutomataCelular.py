import matplotlib.pyplot as plt

def automata_celular(estado_inicial, regla, pasos):
    # Aplica el autómata celular con la regla dada por 'rule' al estado inicial 'initial_state' durante 'steps' iteraciones 
    estado_actual = estado_inicial[:]
    estados = [estado_actual]
    for _ in range(pasos):
        nuevo_estado = [0] * len(estado_actual)
        for i in range(len(estado_actual)):
            # Aplicar la regla a cada célula y su vecindad con vecindad circular
            if i == 0:
                vecindad = (estado_actual[-1], estado_actual[i], estado_actual[i + 1])
            elif i == len(estado_actual) - 1:
                vecindad = (estado_actual[i - 1], estado_actual[i], estado_actual[0])
            else:
                vecindad = (estado_actual[i - 1], estado_actual[i], estado_actual[i + 1])

            nuevo_estado[i] = regla.get(vecindad)

        estado_actual = nuevo_estado[:]
        estados.append(estado_actual)

    return estados

# Ingreso de la matriz para realizar el autómata celular
n_mtrz = 0
print("Ingrese un número impar para la matriz cuadrada para realizar el autómata:")
n_mtrz = int(input())
while n_mtrz%2 == 0:
  print("Número no valido")
  print("Ingrese un número impar para la matriz cuadrada para realizar el autómata:")
  n_mtrz = int(input())

# Definir la regla según la descripción dada
regla = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 1,
    (0, 1, 1): 1,
    (1, 0, 0): 1,
    (1, 0, 1): 0,
    (1, 1, 0): 0,
    (1, 1, 1): 0
}

# Crear un estado inicial con 1 en el centro de la vecindad
estado_inicial = []
for i in range(n_mtrz):
  estado_inicial.append(0)
estado_inicial[int((n_mtrz/2))] = 1
print(estado_inicial)

# Con Matriz Cuadrada
print("Matriz Cuadrada")
# Número de iteraciones (pasos en el tiempo)
pasos = n_mtrz-1

# Aplicar el autómata celular
evolucion = automata_celular(estado_inicial, regla, pasos)

# Convertir la evolución en una matriz para el gráfico
matriz_evolucion = []
for estado in evolucion:
    matriz_evolucion.append(estado)

# Graficar el autómata celular
plt.figure(figsize=(10, 6))
plt.imshow(matriz_evolucion, cmap='binary')
plt.title('Autómata Celular con regla 30')
plt.xlabel('Posición')
plt.ylabel('Tiempo')
plt.show()


# Con Autómata Escalonado
print("Autómata Escalonado")
# Número de iteraciones (pasos en el tiempo)
pasos = int(n_mtrz/2)

# Aplicar el autómata celular
evolucion = automata_celular(estado_inicial, regla, pasos)

# Convertir la evolución en una matriz para el gráfico
matriz_evolucion = []
for estado in evolucion:
    matriz_evolucion.append(estado)

# Graficar el autómata celular
plt.figure(figsize=(10, 6))
plt.imshow(matriz_evolucion, cmap='binary')
plt.title('Autómata Celular con regla 30')
plt.xlabel('Posición')
plt.ylabel('Tiempo')
plt.show()


# Con Autómata 28x31
print("Autómata 28x31")

estado_inicial = []
for i in range(31):
  estado_inicial.append(0)
estado_inicial[16] = 1

# Número de iteraciones (pasos en el tiempo)
pasos = int(27)

# Aplicar el autómata celular
evolucion = automata_celular(estado_inicial, regla, pasos)

# Convertir la evolución en una matriz para el gráfico
matriz_evolucion = []
for estado in evolucion:
    matriz_evolucion.append(estado)

# Graficar el autómata celular
plt.figure(figsize=(10, 6))
plt.imshow(matriz_evolucion, cmap='binary')
plt.title('Autómata Celular con regla 30')
plt.xlabel('Posición')
plt.ylabel('Tiempo')
plt.show()
