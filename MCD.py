import numpy as np
import matplotlib.pyplot as plt



# Personnalisation du graphe
plt.axhline(0, color='gray', linewidth=0.5)  # axe x
plt.axvline(0, color='gray', linewidth=0.5)  # axe y
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title("Affichage du point (0, 0)")

# Afficher
#plt.show()


# Longueurs des bras
L1, L2 = 0.237321, 0.213

positions = []
x, y, theta = L1+L2, 0.0, 0.0

# État initial
q = np.array([0.0, 0.0])
dt = 0.01
T = 5
n_steps = int(T / dt)

# Profils de vitesse articulaire (fixe ici)
dq = np.array([0.2, 0.2])

for _ in range(n_steps):
    q1, q2 = q

    # Jacobienne analytique 3x2
    J = np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1+q2),  L2*np.cos(q1+q2)],
        [1, 1]
    ])

    dx = J @ dq  # vitesse cartésienne

    # Mise à jour de la position (Euler)
    x += dx[0] * dt
    y += dx[1] * dt
    theta += dx[2] * dt

    positions.append((x, y, theta))
    q += dq * dt

# 🔁 Conversion en tableau numpy
positions = np.array(positions)

# Tracé de la trajectoire

plt.plot(positions[:, 1], positions[:, 0], label='Trajectoire de l’effecteur')
plt.plot(0,L1+L2, 'ro', label='Position initiale (L1+L2, 0)')  # 'ro' = point rouge avec un cercle
plt.plot(0,0,"*", label="Origine de la jambe")
plt.xlabel("Y (m)")
plt.ylabel("X (m)")
plt.title("Trajectoire de l'extrémité de la jambe avec dq1 = 0.2 et dq2 = 0.2")
plt.grid(True)
plt.gca().invert_yaxis()
plt.axis('equal')
plt.legend()
plt.show()
