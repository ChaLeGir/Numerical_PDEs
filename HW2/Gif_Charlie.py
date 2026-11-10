from basic_packages import *
import pickle

with open("charlies_results.pkl", "rb") as file:
    results = pickle.load(file)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

###I didn't pickle the boundary values, so we need to reconstruct them from the grid.---------------------------

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

kt = [0.2, -1, 0.4, -0.6, 0.8]

phases = [0.0, 0.8, 1.5, 2.2, 0.4]
wave_types = ["cos", "sin", "cos", "sin", "cos"]
u_prms = (amplitudes, kx, ky, kt, phases, wave_types)

# t_vals = np.linspace(0, 5*np.pi, 150)
# t_vals = np.linspace(0, 1, 10)

###Unpack a pickle.-------------------------------------------------------------

boundary_u_vals = np.array(results["boundary_u_vals"])
x_vals = results["x_vals"]
t_vals = results["t_vals"]
boundary_u_vals = results["boundary_u_vals"]
x_ib_all = results["x_ib_all"]
ib_cond_vals = results["ib_cond_vals"]
u_sol_vals = results["u_sol_vals"]
boundary_vals = get_boundary_values(x_vals)

fig = plt.figure(figsize=(18, 6))

ax = [
    fig.add_subplot(131, projection='3d'),
    fig.add_subplot(132, projection='3d'),
    fig.add_subplot(133)
]

print(u_sol_vals[0])

min_u_sol = np.min(u_sol_vals)
max_u_sol = np.max(u_sol_vals)

###Reconstruct whatever parts of the graph don't need the .pkl.

mines_cmap_vel_u = mines_cmap_vel_u.resampled(256)

grid = np.linspace(a, b, N+1)[1:-1]
X, Y = np.meshgrid(grid, grid, indexing='ij')

im = ax[2].imshow(
    u_sol_vals[0].T,
    cmap=mines_cmap_vel_u,
    origin='lower',
    extent=[a, b, a, b],
    vmin=min_u_sol,
    vmax=max_u_sol
)

xy_ticks = np.arange(0, 2*np.pi + 0.01, np.pi/2)
xy_labels = [
    r'$0$',
    r'$\frac{\pi}{2}$',
    r'$\pi$',
    r'$\frac{3\pi}{2}$',
    r'$2\pi$'
]

for a_i in ax:
    a_i.set_xlim(a, b)
    a_i.set_ylim(a, b)

    a_i.set_xticks(xy_ticks)
    a_i.set_yticks(xy_ticks)

    a_i.set_xticklabels(xy_labels)
    a_i.set_yticklabels(xy_labels)

    a_i.tick_params(axis='both', labelsize=14)

    fig.canvas.draw()

pos = ax[2].get_position(original=False)

cax = fig.add_axes([
    pos.x1 + 0.01,
    pos.y0,
    0.015,
    pos.height
])

cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=14)

def update(frame):

    print(f"Frame: {frame}, Time: {t_vals[frame]:.2f}")

    # Clear only the 3D axes
    for a_i in ax[:2]:
        a_i.cla()

    z = boundary_u_vals[frame]

    vertices = np.column_stack((
        x_ib_all[0, :],
        x_ib_all[1, :],
        ib_cond_vals[frame]
    ))

    ax[0].computed_zorder = False

    polygon = Poly3DCollection(
        [vertices],
        facecolor=blaster_blue,
        alpha=0.8,
        edgecolor='black',
        zorder=5
    )

    ax[0].add_collection3d(polygon)

    ax[0].scatter(
        boundary_vals[0],
        boundary_vals[1],
        z,
        c=dark_blue,
        depthshade=False
    )

    ax[1].plot_surface(
        X, Y, u_sol_vals[frame],
        cmap=mines_cmap_vel_u,
        vmin=min_u_sol,
        vmax=max_u_sol
    )

    # Update existing 2D image
    im.set_data(u_sol_vals[frame].T)

    
    for a_i in ax:
        a_i.set_xlim(a, b)
        a_i.set_ylim(a, b)

        a_i.set_xticks(xy_ticks)
        a_i.set_yticks(xy_ticks)

        a_i.set_xticklabels(xy_labels, fontsize=16)
        a_i.set_yticklabels(xy_labels, fontsize=16)

        if hasattr(a_i, 'set_zlim'):
            a_i.set_zlim(min_u_sol, max_u_sol)
            a_i.set_zticks([0, 1])
            a_i.tick_params(axis='z', labelsize=16)

    # Outside update()
    im.set_data(u_sol_vals[frame].T)

ani = FuncAnimation(
    fig,
    update,
    frames=len(t_vals),
    interval=30,  # Milliseconds between frames
    repeat=True
)

ani.save(
    "immersed_boundary.gif",
    writer=PillowWriter(fps=10)
)

plt.show()