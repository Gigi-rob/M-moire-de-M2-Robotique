import numpy as np
import matplotlib.pyplot as plt

# Paramètres
L1, L2 = 0.237321, 0.213
dt = 0.01
T = 5
n_steps = int(T / dt)

# Initialisation
q = np.array([3.14/2, -3.14/4])
positions = []
q1_list = []
q2_list = []

# Vitesse cartésienne cible
dx_target = np.array([0 , -0.05])

def jacobian(q, L1, L2):
    q1, q2 = q
    return np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1+q2),  L2*np.cos(q1+q2)]
    ])

for _ in range(n_steps):
    J = jacobian(q, L1, L2)
    J_pinv = np.linalg.pinv(J)
    dq = J_pinv @ dx_target  # MCI

    q += dq * dt
    q1_list.append(q[0])
    q2_list.append(q[1])

    x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
    y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    positions.append((x, y))

positions = np.array(positions)

print(f"{L1 * np.cos(1.3) + L2 * np.cos(1.3-1.15)}")
print(f"{L1 * np.sin(1.3) + L2 * np.sin(1.3-1.15)}")

# Trajectoire cartésienne
plt.figure()
plt.plot(positions[:,1], positions[:, 0], label="Trajectoire (MCI)")
plt.plot(positions[0,1], positions[0, 0], 'ro', label="Départ")
plt.plot(0,0,"*", label="Origine de la jambe")
plt.xlabel("Y (m)")
plt.ylabel("X (m)")
plt.title("Trajectoire de l'extrémité de la jambe avec dx = 0 et dy = -0.05 et init = (pi/2 ; -pi/4)")
plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()

# Tracé de q1 et q2
temps = np.linspace(0, T, n_steps)
plt.figure()
plt.plot(temps, q1_list, label="q1 (rad)")
plt.plot(temps, q2_list, label="q2 (rad)")
plt.xlabel("Temps (s)")
plt.ylabel("Angle articulaire (rad)")
plt.grid(True)
plt.legend()
plt.title("Évolution des angles articulaires")
plt.show()

