import sympy as sp

# Déclaration des variables symboliques
q1, q2, L1, L2 = sp.symbols('q1 q2 L1 L2')

# Définition de la Jacobienne symbolique (2x2)
J = sp.Matrix([
    [-L1*sp.sin(q1) - L2*sp.sin(q1 + q2), -L2*sp.sin(q1 + q2)],
    [ L1*sp.cos(q1) + L2*sp.cos(q1 + q2),  L2*sp.cos(q1 + q2)]
])

# Affichage de la Jacobienne
#print("✔️ Jacobienne J(q1, q2, L1, L2) =")
sp.pprint(J)

# Déterminant de la Jacobienne
detJ = J.det()
#print("\n✔️ Déterminant de J =")
sp.pprint(detJ)

# Inverse de la Jacobienne (si elle est inversible)
J_inv = J.inv()
#print("\n✔️ Inverse symbolique de J =")
sp.pprint(J_inv)
