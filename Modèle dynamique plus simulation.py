import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# === Variables symboliques ===
L1, L2, H = sp.symbols('L1 L2 H')
q1, q2, dq1, dq2, ddq1, ddq2 = sp.symbols('q1 q2 dq1 dq2 ddq1 ddq2')
m1, m2, g = sp.symbols('m1 m2 g')

# Matrices d'inertie avec termes croisés (exemple)
I1 = sp.Matrix([[0.02, 0.001, 0.0],
                [0.001, 0.018, 0.0],
                [0.0,   0.0,   0.005]])

I2 = sp.Matrix([[0.015, 0.0005, 0.0],
                [0.0005, 0.012,  0.0],
                [0.0,    0.0,    0.004]])

# Transformations homogènes
M01 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, L1*sp.cos(q1)/2],
                 [sp.sin(q1),  sp.cos(q1), 0, L1*sp.sin(q1)/2],
                 [0,           0,          1, 0],
                 [0,           0,          0, 1]])

M12 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, L2*sp.cos(q2)/2],
                 [sp.sin(q2),  sp.cos(q2), 0, L2*sp.sin(q2)/2],
                 [0,           0,          1, 0],
                 [0,           0,          0, 1]])

MGD = M01 * M12

# Jacobiennes translatoires
Jt1 = sp.Matrix([[sp.diff(M01[0, 3], q1), sp.diff(M01[0, 3], q2)],
                 [sp.diff(M01[1, 3], q1), sp.diff(M01[1, 3], q2)],
                 [0, 0]])

Jt2 = sp.Matrix([[sp.diff(MGD[0, 3], q1), sp.diff(MGD[0, 3], q2)],
                 [sp.diff(MGD[1, 3], q1), sp.diff(MGD[1, 3], q2)],
                 [0, 0]])

# Jacobiennes rotationnelles
Z0 = sp.Matrix([0, 0, 1])
Jr1 = sp.Matrix.hstack(Z0, sp.Matrix([0, 0, 0]))
Jr2 = sp.Matrix.hstack(Z0, Z0)

# Matrice d'inertie D(q)
D = sp.simplify(m1*Jt1.T*Jt1 + m2*Jt2.T*Jt2 + Jr1.T*I1*Jr1 + Jr2.T*I2*Jr2)

# Gravité
H1 = H - L1*sp.cos(q1)/2
H2 = H - L1*sp.cos(q1) - L2*sp.cos(q1+q2)/2
Ep = g*(m1*H1 + m2*H2)
V = sp.Matrix([sp.diff(Ep, q1), sp.diff(Ep, q2)])

# === Calcul complet de C(q,dq) via coefficients de Christoffel ===
q = [q1, q2]
n = 2
C_matrix = sp.zeros(2, 2)
for i in range(n):
    for j in range(n):
        sum_term = 0
        for k in range(n):
            c_ijk = 0.5*(sp.diff(D[i, j], q[k]) +
                         sp.diff(D[i, k], q[j]) -
                         sp.diff(D[j, k], q[i]))
            sum_term += c_ijk * [dq1, dq2][k]
        C_matrix[i, j] = sum_term

# === Lambdify pour D, C et V ===
D_func = sp.lambdify((q1, q2, L1, L2, m1, m2), D, 'numpy')
C_func = sp.lambdify((q1, q2, dq1, dq2, L1, L2, m1, m2), C_matrix, 'numpy')
V_func = sp.lambdify((q1, q2, L1, L2, m1, m2, g, H), V, 'numpy')

# === Paramètres numériques ===
L1_val, L2_val = 0.237321, 0.213
m1_val, m2_val = 1.0, 0.8
g_val, H_val = 9.81, 0.5

# Trajectoire polynomiale 5e ordre
def poly5(t, q0, qf, T):
    a0, a1, a2 = q0, 0, 0
    a3 = 10*(qf-q0)/T**3
    a4 = -15*(qf-q0)/T**4
    a5 = 6*(qf-q0)/T**5
    pos = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    vel = 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    acc = 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return pos, vel, acc

# Simulation
t0, tf, dt = 0, 2, 0.01
time = np.arange(t0, tf+dt, dt)
q1_list, q2_list, tau1, tau2 = [], [], [], []

for t in time:
    q1v, dq1v, ddq1v = poly5(t, 0, np.pi/4, tf)
    q2v, dq2v, ddq2v = poly5(t, 0, -np.pi/8, tf)
    q1_list.append(q1v)
    q2_list.append(q2v)

    D_eval = D_func(q1v, q2v, L1_val, L2_val, m1_val, m2_val)
    C_eval = C_func(q1v, q2v, dq1v, dq2v, L1_val, L2_val, m1_val, m2_val)
    V_eval = V_func(q1v, q2v, L1_val, L2_val, m1_val, m2_val, g_val, H_val)

    ddq = np.array([ddq1v, ddq2v])
    dq = np.array([dq1v, dq2v])

    tau = D_eval @ ddq + C_eval @ dq + V_eval.flatten()
    tau1.append(tau[0])
    tau2.append(tau[1])

# Affichage
plt.figure()
plt.plot(time, tau1, label="τ1 (N·m)")
plt.plot(time, tau2, label="τ2 (N·m)")
plt.title("Couples articulaires (Euler-Lagrange complet)")
plt.xlabel("Temps (s)")
plt.ylabel("Couple (N·m)")
plt.legend()
plt.grid()

plt.figure()
plt.plot(time, q1_list, label="q1 (rad)")
plt.plot(time, q2_list, label="q2 (rad)")
plt.title("Trajectoires articulaires")
plt.xlabel("Temps (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()
