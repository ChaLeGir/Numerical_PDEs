from basic_packages import *
from svgpathtools import svg2paths
from scipy.sparse.linalg import gmres, LinearOperator
import random
from shapely.geometry import Point, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse.linalg import splu
from scipy.linalg import solve
import pickle


###DEFINED FUNCTIONS####--------------------------------------------------------------------

def sample_and_compact(x, y, N):
    assert x.shape == y.shape, "x and y must have the same shape"
    assert N > 0, "N must be positive"
    assert N <= len(x), "N must be less than or equal to the length of x and y"
    indices = np.linspace(0, len(x)-1, N, dtype=int)
    x_sampled = x[indices]
    y_sampled = y[indices]
    return np.array([x_sampled, y_sampled])

def u_bdy(x, t):
    return 1 + .2*((x[0]-np.pi)*np.cos(t)
                + (x[1]-np.pi)*np.sin(t))
    # return 1.0

def u(x, t, amplitudes, kx, ky, kt, phases, wave_types):
    result = np.zeros_like(np.asarray(x[0]) + np.asarray(x[1]), dtype=float)

    for j in range(5):
        theta = kx[j]*x[0] + ky[j]*x[1] + kt[j]*t + phases[j]
        trig = np.sin if wave_types[j] == "sin" else np.cos

        result += amplitudes[j] * trig(theta)

    return result

    # return 0.0

def f(x, t, amplitudes, kx, ky, kt, phases, wave_types):
    """Spatial Laplacian of u: f = u_xx + u_yy."""
    result = np.zeros_like(np.asarray(x[0]) + np.asarray(x[1]), dtype=float)

    for j in range(5):
        theta = kx[j]*x[0] + ky[j]*x[1] + kt[j]*t + phases[j]
        trig = np.sin if wave_types[j] == "sin" else np.cos

        result -= (
            amplitudes[j]
            * (kx[j]**2 + ky[j]**2)
            * trig(theta)
        )

    return result
    # return 0.0

def f_flat(x):
    return 0

def u_ibd_flat(x):
    return 0

def delta_a(x, y, a):
    exp_term = np.exp(-(x**2 + y**2)/(2 * a**2))
    return exp_term / (2 * np.pi * a**2)

def get_J_ib(N, N_ib, N_ib_edges, x_vals, a, x_ib_vals):
    J_ib = np.zeros((x_ib_vals.shape[1], (N-1)**2))
    for k in range(x_ib_vals.shape[1]):
        for i in range(1, N):
            for j in range(1, N):
                x = [x_vals[0, i], x_vals[1, j]]
                x_ib = x_ib_vals[:, k]
                delta_result = delta_a(x[0]-x_ib[0], x[1]-x_ib[1], a)
                if delta_result >= 1e-4:
                    J_ib[k, get_interior_index(i, j, N)] = delta_result
    return J_ib

def get_S_ib(N, N_ib, N_ib_edges, x_vals, a, x_ib_vals):

    S_ib = np.zeros(((N-1)**2, x_ib_vals.shape[1]))
    print(f"I have {S_ib.shape[0]} rows to create.")
    print(f"My a is {a}, and h is { (b-a)/N }")
    print(f"x_ib_vals.shape = {x_ib_vals.shape}")
    print(f"x_ib_vals.shape[1] = {x_ib_vals.shape[1]}")
    print(f"Total number of IB points including edges: {x_ib_vals.shape[1]}")
    for i in range(1, N):
        for j in range(1, N):
            for k in range(x_ib_vals.shape[1]):
                x = [x_vals[0, i], x_vals[1, j]]
                x_ib = x_ib_vals[:, k]
                delta_result = delta_a(x[0]-x_ib[0], x[1]-x_ib[1], a)
                if delta_result >= 1e-4:
                    S_ib[get_interior_index(i, j, N), k] = delta_result
                sampled_count = 0
                if random.random() < 0.001:
                    print(f"Randomly Sampled Delta {sampled_count} result", S_ib[get_interior_index(i, j, N), k])
                    sampled_count += 1
            # print(f"Processed interior index {get_interior_index(i, j, N)} of {(N-1)**2} for all k.")
    print(f"Finished creating S_ib with shape {S_ib.shape}")
    return S_ib

def get_immersed_boundary_u(u, N_bd, x_ib_vals):
    u_ib = np.zeros(N_bd)
    for k in range(N_bd):
        u_ib[k] = u([x_ib_vals[0, k], x_ib_vals[1, k]])
    return u_ib

def get_J_ib_fast(u_approx, S_ib, h):
    return h**2 * (S_ib.T @ u_approx)

#Let z be the concatenated u and q vectors.
def LHS(z_approx, N, h, A, S_ib, J_ib):
    # print(f"LHS called with z_approx of length {len(z_approx)}")
    assert len(z_approx) == (N-1)**2 + S_ib.shape[1], "Length of z_approx does not match expected size."
    u_approx = z_approx[:(N-1)**2]
    q_approx = z_approx[(N-1)**2:]

    f = A @ u_approx - S_ib @ q_approx
    u_ib = h**2 * (J_ib @ u_approx)

    return np.concatenate([f, u_ib])

def RHS(N, h, x, u, f, u_intbdy, u_intbdy_outer, u_intbdy_inner, x_ib, x_ib_outer, x_ib_inner):
    #This is the original rhs, or f modified fot Dirichilet boundary conditions.
    print(f"Made a ")
    B_bd = make_B_bd(N, h)
    u_bd = make_u_bd(N, u, x)
    f_vec = make_f_vec(N, f, x)
    b = make_rhs(f_vec, B_bd, u_bd)

    #Next we make the interior boundary rhs constraint.
    u_ib = get_immersed_boundary_u(u_intbdy, x_ib.shape[1], x_ib)
    u_ib_outer = get_immersed_boundary_u(u_intbdy_outer, x_ib_outer.shape[1], x_ib_outer)
    u_ib_inner = get_immersed_boundary_u(u_intbdy_inner, x_ib_inner.shape[1], x_ib_inner)

    return np.concatenate([b, u_ib, u_ib_outer, u_ib_inner])

    #Reskin of HW0 Finite difference code. 
#Except this time we will put everything inside defined functions for ease of use.

def make_grid(a, b, N):
    lnspc = np.linspace(a, b, N+1)
    return np.row_stack((lnspc, lnspc))

def get_interior_index(i, j, N):    
    assert 1 <= i < N
    assert 1 <= j < N
    return (i-1)*(N-1) + (j-1)

def get_boundary_index(i, j, N):
    assert i == 0 or i == N or j == 0 or j == N, "Point is not on the boundary"
    if i == 0:
        return j  # Top boundary
    if i == N:
        return N + j + 1  # Bottom boundary
    if j == 0:
        return 2*N + i + 1 # Left boundary
    if j == N:
        return 3*N + i # Right boundary

def populate_A(A, N):
    for i in range(1, N):
        for j in range(1, N):
            A[get_interior_index(i, j, N), get_interior_index(i, j, N)] = -4    #Self-reference term
            if i > 1:
                A[get_interior_index(i, j, N), get_interior_index(i-1, j, N)] = 1 #Above neighbor compressed to row-major order
            if i < N-1:
                A[get_interior_index(i, j, N), get_interior_index(i+1, j, N)] = 1 #Below neighbor compressed to row-major order
            if j > 1:
                A[get_interior_index(i, j, N), get_interior_index(i, j-1, N)] = 1 #Left neighbor compressed to row-major order
            if j < N-1:
                A[get_interior_index(i, j, N), get_interior_index(i, j+1, N)] = 1 #Right neighbor compressed to row-major order
    return A

def populate_B_bd(B_bd, N):
    for i in range(1, N):
        for j in range(1, N):
            if i == 1:
                B_bd[(i-1)*(N-1) + (j-1), get_boundary_index(i-1, j, N)] = 1 #Top boundary compressed to row-major order. Note that the index of the boundary matrix is offset from the interior point matrix. 
            if i == N-1:
                B_bd[(i-1)*(N-1) + (j-1), get_boundary_index(i+1, j, N)] = 1 #Bottom boundary compressed to row-major order.
            if j == 1:
                B_bd[(i-1)*(N-1) + (j-1), get_boundary_index(i, j-1, N)] = 1 #Left boundary compressed to row-major order.
            if j == N-1:
                B_bd[(i-1)*(N-1) + (j-1), get_boundary_index(i, j+1, N)] = 1 #Right boundary compressed to row-major order.
    return B_bd

def make_A(N, h):
    A = sp.sparse.lil_matrix(((N-1)**2, (N-1)**2))
    A = populate_A(A, N)
    return A.tocsr() * (1/h**2)

def make_B_bd(N, h):
    B_bd = sp.sparse.lil_matrix(((N-1)**2, 4*N))
    B_bd = populate_B_bd(B_bd, N)
    return B_bd.tocsr() * (1/h**2)

def make_u_bd(N, u, x):
    u_bd = np.zeros(4*N)
    for i in range(N+1):
        for j in range(N+1):
            if i == 0 or i == N or j == 0 or j == N:
                u_bd[get_boundary_index(i, j, N)] = u([x[0][i], x[1][j]])
    return u_bd

def make_f_vec(N, f, x):
    f_vec = np.zeros((N-1)**2)
    for i in range(1, N):
        for j in range(1, N):
            f_vec[get_interior_index(i, j, N)] = f([x[0][i], x[1][j]])
    return f_vec

def make_rhs(f_vec, B_bd, u_bd):
    return f_vec - B_bd @ u_bd

def full_system(N, a, b, u, f):
    h = (b-a) / N
    x = make_grid(a, b, N)
    A = make_A(N, h)
    B_bd = make_B_bd(N, h)
    u_bd = make_u_bd(N, u, x)
    f_vec = make_f_vec(N, f, x)
    b = make_rhs(f_vec, B_bd, u_bd)
    u_approx = sp.sparse.linalg.spsolve(A, b)
    return u_approx

def get_exact(N, a, b, u):
    x = make_grid(a, b, N)
    u_exact = np.zeros((N-1)**2)
    for i in range(1, N):
        for j in range(1, N):
            u_exact[get_interior_index(i, j, N)] = u([x[0][i], x[1][j]])
    return u_exact

def test_ib(u, u_intbdy, f, N, a, b, x_ib_vals, reg_var):
    print("Starting test_ib")
    x_vals = make_grid(a, b, N)
    h = (b-a) / N
    A = make_A(N, h)
    S_ib = get_S_ib(x_vals, reg_var*h, x_ib_vals)
    print("S_ib shape:", S_ib.shape)
    print("S_ib sample row:", S_ib[0])
    print("A shape:", A.shape)
    print("A sample row:", A[0])
    right_hand_side = RHS(N, a, b, u, u_intbdy, f, x_ib_vals)
    print("Right hand side shape:", right_hand_side.shape)

    print("Flag 1")

    N_ib = x_ib_vals.shape[1]

    print("Flag 2")

    A_op = LinearOperator(
    shape=((N-1)**2 + N_ib, (N-1)**2 + N_ib),
    matvec=lambda z: LHS(z, N, h, A, S_ib),
    dtype=float
    )

    print("Flag 3")

    solution, info = gmres(
        A_op,
        right_hand_side,
        restart=1000,
        rtol=1e-8,
        atol=0,
        maxiter=1000
    )

    print("Flag 4")

    return solution, info

def wrap_u(u, t, u_prms):
    print("Wrapping u with t =", t)
    return lambda coords: u(coords, t, *u_prms)

def wrap_u_bdy(u_bdy, t):
    print("Wrapping u_bdy with t =", t)
    return lambda coords: u_bdy(coords, t)

def wrap_f(f, t, u_prms):
    print("Wrapping f with t =", t)
    return lambda coords: f(coords, t, *u_prms)

def get_boundary_values(x):
    X, Y = np.meshgrid(x[0], x[1], indexing='ij')

    # Extract all four edges
    boundary_mask = np.zeros(X.shape, dtype=bool)
    boundary_mask[[0, -1], :] = True
    boundary_mask[:, [0, -1]] = True

    # Return coordinates in your preferred (2, M) format
    x_boundary = np.array([X[boundary_mask], Y[boundary_mask]])
    return x_boundary

###PARAMETERS####-------------------------------------------------------------------------------
N = 200 # Resolution of finite difference grid
a = 0 
b = 2 * np.pi
h = (b - a) / N
N_ib = 200 # Number of blaster immersed boundary points
N_ib_edges = 100
Scaling_Factor = 0.15
Blaster_Offset = 0.25

amplitudes = [.2, 0.1, 0.04, 0.05, 0.15]

kx = [3, 1.0, 0, 0.2, 0.4]
ky = [0, 2.0, -1.0, 0.7, -0.1]

kt = [4.2, -1, 0.4, -0.6, 0.8]

phases = [0.0, 0.8, 1.5, 2.2, 0.4]
wave_types = ["cos", "sin", "cos", "sin", "cos"]
u_prms = (amplitudes, kx, ky, kt, phases, wave_types)

t_vals = np.linspace(0, 10*np.pi, 500)

###LOAD AND PROCESS SVG####-------------------------------------------------------------------

paths, attributes = svg2paths("colorado-mines.svg")

path = paths[0]

theta = np.linspace(0, 1, 500, endpoint=False)
points = np.array([path.point(ti) for ti in theta])

x = points.real
y = points.imag
plt.figure(num="Original Blaster")
plt.plot(x, y)
plt.show(block=False)

current_min = min(np.min(x), np.min(y))
current_max = max(np.max(x), np.max(y))

desired_min = Scaling_Factor * 2* np.pi
desired_max = (1 - Scaling_Factor) * 2* np.pi

scale = (desired_max - desired_min) / (current_max - current_min)
x_scaled = desired_min + (x - current_min) * scale
y_scaled = desired_min + (y - current_min) * scale


plt.figure(num="My Scaled Blaster")
plt.plot(x_scaled, y_scaled)
plt.show(block=False)

x_ib = sample_and_compact(x_scaled, y_scaled, N_ib)

plt.figure(num="Sampled Blaster")
plt.plot(x_ib[0], x_ib[1], 'o')
plt.show(block=False)

# points = np.column_stack((x_ib[0], x_ib[1]))
# polygon = Polygon(points)
# outer_polygon = polygon.buffer(Blaster_Offset)
# x_outer, y_outer = outer_polygon.exterior.xy
# x_ib_outer = sample_and_compact(np.array(x_outer), np.array(y_outer), N_ib_edges)
x_ib_outer = np.array([[],[]])

# inner_polygon = polygon.buffer(-Blaster_Offset)
# if inner_polygon.geom_type == 'Polygon':
#     inner_polygons = [inner_polygon]
# elif inner_polygon.geom_type == 'MultiPolygon':
#     inner_polygons = list(inner_polygon.geoms)
# else:
#     inner_polygons = []
# x_ib_inner_x = np.array([])
# x_ib_inner_y = np.array([])
# for p in inner_polygons:
#     x_inner, y_inner = p.exterior.xy
#     x_ib_inner_x = np.concatenate((x_ib_inner_x, x_inner))
#     x_ib_inner_y = np.concatenate((x_ib_inner_y, y_inner))
# x_ib_inner = np.array([x_ib_inner_x, x_ib_inner_y]) if x_ib_inner_x.size < N_ib_edges else sample_and_compact(x_ib_inner_x, x_ib_inner_y, N_ib_edges)
# if x_ib_inner_x.size > 0:
#     x_ib_inner = sample_and_compact(x_ib_inner_x, x_ib_inner_y, N_ib_edges)
x_ib_inner = np.array([[],[]])

print("x_ib_outer shape:", x_ib_outer.shape)
print("x_ib_inner shape:", x_ib_inner.shape)
print("x_ib shape:", x_ib.shape)
plt.figure(num="Inner and Outer Blaster")
plt.plot(x_ib_outer[0, :], x_ib_outer[1, :], '-', label='Outer Blaster', c='red')
if x_ib_inner.size > 0:
    plt.plot(x_ib_inner[0, :], x_ib_inner[1, :], '-', label='Inner Blaster', c='blue')
plt.plot(x_ib[0, :], x_ib[1, :], '-', label='Original Blaster', c='green')
plt.legend()
plt.show(block=False)

# solution, info = test_ib(wrap_u(u, t_vals[0], u_prms), wrap_u_bdy(u_bdy, t_vals[0]), wrap_f(f, t_vals[0], u_prms), N, a, b, x_ib, 1.2*h)
# Going through test_ib to fix problems. The following is in test_ib:
print("Starting test_ib")
x_vals = make_grid(a, b, N)

A = make_A(N, h)
print("A shape:", A.shape)
print("A sample row:", A[100])

print("Data type for x_ib:", type(x_ib))
if type(x_ib) == list:
    x_ib = np.array(x_ib)
print("x_ib shape:", x_ib.shape)
x_ib_all = np.hstack([x_ib, x_ib_outer, x_ib_inner])
print("x_ib_all shape:", x_ib_all.shape)
S_ib = get_S_ib(N, N_ib, N_ib_edges, x_vals, 1.2*h, x_ib_all)
print("S_ib shape:", S_ib.shape)
print("S_ib sample row:", S_ib[0])
print("Integral of S_ib over interior points:", h**2 * np.sum(S_ib))
J_ib = get_J_ib(N, N_ib, N_ib_edges, x_vals, 1.2*h, x_ib_all)
print("J_ib shape:", J_ib.shape)
J = h**2 * J_ib

# sample_t_index = 5
# wrapped_u = wrap_u(u, t_vals[sample_t_index], u_prms)
# wrapped_f = wrap_f(f, t_vals[sample_t_index], u_prms)
# wrapped_u_bdy = wrap_u_bdy(u_bdy, t_vals[sample_t_index])
# u_exact = get_exact(N, a, b, wrapped_u)
# f_exact = get_exact(N, a, b, wrapped_f)
# u_bdy_exact = get_exact(N, a, b, wrapped_u_bdy)
# X, Y = np.meshgrid(np.linspace(a, b, N-1), np.linspace(a, b, N-1))
# Z_u_exact = u_exact.reshape((N-1, N-1))
# Z_f_exact = f_exact.reshape((N-1, N-1))
# Z_u_bdy_exact = u_bdy_exact.reshape((N-1, N-1))
# fig = plt.figure(num="Exact Solutions", figsize = (10, 8))
# ax1 = fig.add_subplot(131, projection='3d')
# ax1.plot_surface(X, Y, Z_u_exact, cmap='viridis')
# ax1.set_title("u_exact")
# ax2 = fig.add_subplot(132, projection='3d')
# ax2.plot_surface(X, Y, Z_f_exact, cmap='viridis')
# ax2.set_title("f_exact")
# ax3 = fig.add_subplot(133, projection='3d')
# ax3.plot_surface(X, Y, Z_u_bdy_exact, cmap='viridis')
# ax3.set_title("u_bdy_exact")
# plt.show(block=False)

n = (N-1)**2
x_ib_all = np.hstack([x_ib, x_ib_outer, x_ib_inner])
grid = np.linspace(a, b, N+1)[1:-1]
#These are the Schur Decomposition steps I only need to do once.
# Factorize the sparse Laplacian once
A_lu = splu(A.tocsc())
# Solve A W = S, one column per immersed-boundary point
W = A_lu.solve(S_ib)
# Construct the Schur complement
H = J @ W
u_sol_vals = []
ib_cond_vals = []
boundary_u_vals = []
boundary_vals = get_boundary_values(x_vals)

for t in t_vals:
    wrapped_u = wrap_u(u, t, u_prms)
    wrapped_f = wrap_f(f, t, u_prms)
    wrapped_u_bdy = wrap_u_bdy(u_bdy, t)


    print("Ready to attempt a construction of the right-hand side!")
    right_hand_side = RHS(
        N, h, x_vals,
        wrapped_u,
        f_flat,
        wrapped_u_bdy,
        u_ibd_flat,
        wrapped_u_bdy,
        x_ib,
        x_ib_outer,
        x_ib_inner,
        )
    print("Right-hand side constructed, with shape:", right_hand_side.shape)
    print("This makes sense, because it should have ", (N-1)**2 + x_ib.shape[1] + x_ib_outer.shape[1] + x_ib_inner.shape[1], " entries.")

# print("Attempting to run a test on the LHS operator.")
# print("Generating z value for testing the LHS operator")
# print("u exact shape: ", u_exact.shape)
# z = np.concatenate([u_exact, np.ones(N_ib)])
# print("z shape: ", z.shape)
# print("Attempting to apply LHS operator to z")
# lhs_result = LHS(z, N, h, A, S_ib, J_ib)
# print("LHS result shape: ", lhs_result.shape)
# print("Expected LHS result shape: ", (N-1)**2 + N_ib)

##END OF WHAT CURRENTLY WORKS; THE FOLLOWING IS BEING BUILT AND DEBUGGED.
##IF IT WORKS, PASTE IT ABOVE THIS LINE!

# D = (N-1)**2 + N_ib + 2*N_ib_edges
# A_op = LinearOperator(
#     shape=(D, D),
#     matvec=lambda z: LHS(z, N, h, A, S_ib, J_ib),
#     dtype=np.float64
# )

# solution, info = gmres(
#     A_op,
#     right_hand_side,
#     restart=1000,
#     rtol=1e-8,
#     atol=0,
#     maxiter=1000
# )

# print("GMRES info: ", info)
# print("GMRES solution shape: ", solution.shape)

    
    f_rhs = right_hand_side[:n]
    g_rhs = right_hand_side[n:]


    # Solve A w = f
    w = A_lu.solve(f_rhs)


    # Solve for immersed-boundary forces
    rhs_q = g_rhs - J @ w
    print("Solving for immersed-boundary forces...")
    q = solve(H, rhs_q, assume_a='sym')
    # Recover the solution on the Cartesian grid
    print("Recovering the solution on the Cartesian grid...")
    u_sol_vec = w + W @ q

    u_sol = u_sol_vec.reshape((N-1, N-1))
    u_sol_vals.append(u_sol)
    ib_cond_vals.append(right_hand_side[n:])

    boundary_u = [wrapped_u([boundary_vals[0, i], boundary_vals[1, i]]) for i in range(boundary_vals.shape[1])]
    boundary_u_vals.append(boundary_u)


fig = plt.figure(num="GMRES Solution", figsize = (10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(grid, grid, indexing='ij')
ax.plot_surface(X, Y, u_sol_vals[-1], cmap='viridis')
ax.set_title("u_sol")
vertices = np.column_stack((
    x_ib_all[0, :],
    x_ib_all[1, :],
    ib_cond_vals[-1]
))
plt.show(block=False)


polygon_2 = Poly3DCollection(
    [vertices],
    facecolor='red',
    alpha=0.3,
    edgecolor='black'
)
fig = plt.figure(num="Immersed Boundary Points Only", figsize = (10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(polygon_2)
ax.scatter(boundary_vals[0], boundary_vals[1], boundary_u_vals[-1], c='blue')
plt.show(block=False)

# print("x_ib shape: ", x_ib.shape),
# print("z_ib_vals shape: ", z_ib_vals.shape)
# print(z_ib_vals)

# print("Residual:")
# residual = right_hand_side - LHS(u_sol_vec, N, h, A, S_ib, J_ib)
# print("Residual shape: ", residual.shape)
# print("Residual norm: ", np.linalg.norm(residual))
# print(residual)

# print("Confirmation that the interpolation at immersed boundary points is consistent with the solution:")
# interpolated_values = (J_ib @ u_sol_vec[:(N-1)**2]) * h**2
# print("Interpolated values shape: ", interpolated_values.shape)
# print("Interpolated values residual norm: ", np.linalg.norm(interpolated_values - z_ib_vals))

###END OF CODE, WAIT FOR INPUT###------------------------------------------------

input("Press Enter to close all figures and continue...")

plt.close('all')

results = {
    "x_vals": x_vals,
    "t_vals": t_vals,
    "x_ib_all": x_ib_all,
    "u_sol_vals": u_sol_vals,
    "ib_cond_vals": ib_cond_vals,
    "boundary_u_vals": boundary_u_vals,
    "grid": grid,
    "N": N,
    "N_ib": N_ib,
    "a": a,
    "b": b,
    "h": h
}

with open("charlies_results.pkl", "wb") as file:
    pickle.dump(results, file)