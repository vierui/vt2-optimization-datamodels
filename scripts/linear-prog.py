from scipy.optimize import linprog

# Define cost vector (cost coefficients for wind and solar for all hours)
c = []
for t in range(24):
    # Wind (15), Solar (5), Load (0), Free Bus (0), Angles (0, 0, 0, 0)
    c += [15, 5, 0, 0, 0, 0, 0, 0]
    
# Define equality constraints (power balance + reference bus angle constraint)
A_eq = []  # Matrix representing power balance and theta_1 constraints
b_eq = []  # Demand and reference angle vector

# Example: theta_1 constraint (theta_1(t) = 0)
for t in range(24):
    theta_1_constraint = [0, 0, 0, 0, 1, 0, 0, 0]  # Only 1 for theta_1(t), rest are 0
    A_eq.append(theta_1_constraint)  # Add for each hour
    b_eq.append(0)  # Reference angle = 0

# Define inequality constraints (generation limits for wind and solar)
A_ineq = []  # Matrix encoding generation limits
b_ineq = []  # Upper and lower bounds for generation

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq)

# Check and extract the solution
if result.success:
    print("Optimization successful!")
    solution = result.x  # Optimal values of power injections and angles
else:
    print("Optimization failed:", result.message)
