# MTF073 Computational Fluid Dynamics
# Task 1: 2D diffusion equation
# Template prepared by:
# Gonzalo Montero Villar
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# villar@chalmers.se
# November 2020

# Packages needed
import numpy as np
import matplotlib.pyplot as plt


# ===================== Schematic ==================
#
#                  0----------------0
#                  |                |
#                  |                |
#                  |    [i,j+1]     |
#                  |       X        |
#                  |      (N)       |
#                  |                |
#                  |                |
# 0----------------0----------------0----------------0
# |                |                |                |
# |                |                |                |
# |    [i-1,j]     |     [i,j]      |    [i+1,j]     |
# |       X        |       X        |       X        |
# |      (W)       |      (P)       |      (E)       |
# |                |                |                |
# |                |                |                |
# 0----------------0----------------0----------------0
#                  |                |
#                  |                |
#                  |    [i,j-1]     |
#                  |       X        |
#                  |      (S)       |
#                  |                |
#                  |                |
#                  0----------------0
#
# X:  marks the position of the nodes, which are the centers
#     of the control volumes, where temperature is computed.
# 0:  marks the position of the mesh points or control volume
#     corners.
# []: in between square brakets the indexes used to find a 
#     node with respect to the node "P" are displayed.
# (): in between brakets the name given to refer to the nodes
#     in the lectures as well as in the book with respect to 
#     the node "P" are displayed.


def dirichlet_function(x, y, boundaries):
    # calculates boundary condition variables
    i = x
    j = y
    coeffsT[i, j, 0] = (k[i + 1, j] * dx_CV[i + 1, j] / (2 * dxe_N[i, j]) + k[i, j] * dx_CV[i, j] / (2 * dxe_N[i, j])) * \
                       dy_CV[i, j] / dxe_N[i, j]
    coeffsT[i, j, 1] = (k[i - 1, j] * dx_CV[i - 1, j] / (2 * dxw_N[i, j]) + k[i, j] * dx_CV[i, j] / (2 * dxw_N[i, j])) * \
                       dy_CV[i, j] / dxw_N[i, j]
    coeffsT[i, j, 2] = (k[i, j + 1] * dy_CV[i, j + 1] / (2 * dyn_N[i, j]) + k[i, j] * dy_CV[i, j] / (2 * dyn_N[i, j])) * \
                       dx_CV[i, j] / dyn_N[i, j]
    coeffsT[i, j, 3] = (k[i, j - 1] * dy_CV[i, j - 1] / (2 * dys_N[i, j]) + k[i, j] * dy_CV[i, j] / (2 * dys_N[i, j])) * \
                       dx_CV[i, j] / dys_N[i, j]
    for boundary in boundaries:
        coeffsT[x, y, boundary] = 0
        if boundary == 0:
            kb = k[x + 1, y]
            S_P[x, y] -= kb * dy_CV[x, y] / dxe_N[x, y]
            S_U[x, y] += T2 * kb * dy_CV[x, y] / dxe_N[x, y]
        elif boundary == 1:
            continue  # neumann
            #kb = k[x - 1, y]
            #S_P[x, y] -= kb * dy_CV[x, y] / dxw_N[x, y]
            #S_U[x, y] += 5 * kb * dy_CV[x, y] / dxw_N[x, y]
        elif boundary == 2:
            kb = k[x, y + 1]
            S_P[x, y] -= kb * dx_CV[x, y] / dyn_N[x, y]
            S_U[x, y] += T[x, y + 1] * kb * dx_CV[x, y] / dyn_N[x, y]
        elif boundary == 3:
            kb = k[x, y - 1]
            S_P[x, y] -= kb * dx_CV[x, y] / dys_N[x, y]
            S_U[x, y] += T1 * kb * dx_CV[x, y] / dys_N[x, y]
    coeffsT[x, y, 4] = coeffsT[x, y, 0] + coeffsT[x, y, 1] + coeffsT[x, y, 2] + coeffsT[x, y, 3] - S_P[x, y]


# ===================== Inputs =====================

# Geometric inputs

mI = 20  # number of mesh points X direction.
mJ = 20  # number of mesh points Y direction.
grid_type = 'equidistant'  # this sets equidistant mesh sizing or non-equidistant
xL = 1  # length of the domain in X direction
yL = 7  # length of the domain in Y direction
c1 = 25
c2 = 0.25
T1 = 10
T2 = 20

# Solver inputs

nIterations = 10000  # maximum number of iterations
resTolerance = 0.001  # convergence criteria for residuals each variable

# ====================== Code ======================

# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate (check Schematix above)

# Allocate all needed variables
nI = mI + 1  # number of nodes in the X direction. Nodes
# added in the boundaries
nJ = mJ + 1  # number of nodes in the Y direction. Nodes
# added in the boundaries
coeffsT = np.zeros((nI, nJ, 5))  # coefficients for temperature
# E, W, N, S and P
S_U = np.zeros((nI, nJ))  # source term for temperature
S_P = np.zeros((nI, nJ))  # source term for temperature
T = np.zeros((nI, nJ))  # temperature matrix
k = np.zeros((nI, nJ))  # coefficient of conductivity
q = np.zeros((nI, nJ, 2))  # heat flux, first x and then y component

residuals = []  # List containing the value of the residual for each iteration

# Generate mesh and compute geometric variables

# Allocate all variables matrices
xCoords_M = np.zeros((mI, mJ))  # X coords of the mesh points
yCoords_M = np.zeros((mI, mJ))  # Y coords of the mesh points
xCoords_N = np.zeros((nI, nJ))  # X coords of the nodes
yCoords_N = np.zeros((nI, nJ))  # Y coords of the nodes
dxe_N = np.zeros((nI, nJ))  # X distance to east node
dxw_N = np.zeros((nI, nJ))  # X distance to west node
dyn_N = np.zeros((nI, nJ))  # Y distance to north node
dys_N = np.zeros((nI, nJ))  # Y distance to south node
dx_CV = np.zeros((nI, nJ))  # X size of the control volume
dy_CV = np.zeros((nI, nJ))  # Y size of the control volume

if grid_type == 'equidistant':
    # Control volume size
    dx = np.ones(nI) * xL / (mI - 1)
    dy = np.ones(nJ) * yL / (mJ - 1)
elif grid_type == 'non-equidistant':
    # first and last value must add to 2 (any combination works)
    # size of cells will way with x and y according to the used values
    dx = np.linspace(0.1, 1.9, nI) * xL / (mI - 1)
    dy = np.linspace(1.6, 0.4, nJ) * yL / (mJ - 1)

for i in range(mI):
    for j in range(mJ):
        # For the mesh points
        if i > 0:
            xCoords_M[i, j] = xCoords_M[i - 1, j] + dx[i]
        if j > 0:
            yCoords_M[i, j] = yCoords_M[i, j - 1] + dy[j]

            # For the nodes
        if i > 0:
            xCoords_N[i, j] = 0.5 * (xCoords_M[i, j] + xCoords_M[i - 1, j])
        if i == (mI - 1) and j > 0:
            yCoords_N[i + 1, j] = 0.5 * (yCoords_M[i, j] + yCoords_M[i, j - 1])
        if j > 0:
            yCoords_N[i, j] = 0.5 * (yCoords_M[i, j] + yCoords_M[i, j - 1])
        if j == (mJ - 1) and i > 0:
            xCoords_N[i, j + 1] = 0.5 * (xCoords_M[i, j] + xCoords_M[i - 1, j])

            # Fill dx_CV and dy_CV
        if i > 0:
            dx_CV[i, j] = xCoords_M[i, j] - xCoords_M[i - 1, j]
        if j > 0:
            dy_CV[i, j] = yCoords_M[i, j] - yCoords_M[i, j - 1]

xCoords_N[-1, :] = xL
yCoords_N[:, -1] = yL

# Fill dxe, dxw, dyn and dys
dxe_N[0:nI - 1, :] = np.diff(xCoords_N, axis=0)
dxw_N[1:nI, :] = np.diff(xCoords_N, axis=0)
dyn_N[:, 0:nJ - 1] = np.diff(yCoords_N)
dys_N[:, 1:nJ] = np.diff(yCoords_N)

# Initialize variable matrices and boundary conditions
T[-1, :] = T2  # boundary 2
T[:, 0] = T1  # boundary 1
#T[0,:] = 5 # boundary 4 when changed from neumann

# boundary 3
for i in range(nI):
    T[i, -1] = 5 + 3 * (1 + 5 * xCoords_N[i, -1] / xL)

# Looping
x = []
for iter in range(nIterations):
    # Update conductivity coefficient matrix, k, according to your case
    k = (16 * (yCoords_N/yL + 30 * T/T1))*100
    S_P = -c2 * 15 * T * dx_CV * dy_CV
    S_U = c1 * 15 * dx_CV * dy_CV

    # Compute coeffsT for all the nodes which are not boundary nodes
    # compute boundary coefficients
    for i in range(2, nI - 2):
        j = 1
        dirichlet_function(i, j, boundaries=[3])

        j = nJ - 2
        dirichlet_function(i, j, boundaries=[2])

    for j in range(2, nJ - 2):
        i = 1
        dirichlet_function(i, j, boundaries=[1])

        i = nI - 2
        dirichlet_function(i, j, boundaries=[0])

    ## Compute coefficients for inner nodes
    for i in range(2, nI - 2):
        for j in range(2, nJ - 2):
            # k is calculated as (k of the node)*(half the size of the other cell)/(length between nodes).
            coeffsT[i, j, 0] = (k[i + 1, j] * dx_CV[i, j] / (2 * dxe_N[i, j]) + k[i, j] * dx_CV[i + 1, j] / (
                        2 * dxe_N[i, j])) * dy_CV[i, j] / dxe_N[i, j]
            coeffsT[i, j, 1] = (k[i - 1, j] * dx_CV[i, j] / (2 * dxw_N[i, j]) + k[i, j] * dx_CV[i - 1, j] / (
                        2 * dxw_N[i, j])) * dy_CV[i, j] / dxw_N[i, j]
            coeffsT[i, j, 2] = (k[i, j + 1] * dy_CV[i, j] / (2 * dyn_N[i, j]) + k[i, j] * dy_CV[i, j + 1] / (
                        2 * dyn_N[i, j])) * dx_CV[i, j] / dyn_N[i, j]
            coeffsT[i, j, 3] = (k[i, j - 1] * dy_CV[i, j] / (2 * dys_N[i, j]) + k[i, j] * dy_CV[i, j - 1] / (
                        2 * dys_N[i, j])) * dx_CV[i, j] / dys_N[i, j]
            coeffsT[i, j, 4] = coeffsT[i, j, 0] + coeffsT[i, j, 1] + coeffsT[i, j, 2] + coeffsT[i, j, 3] - S_P[i, j]
    ## corner nodes
    i = 1
    j = 1
    dirichlet_function(i, j, boundaries=[1, 3])

    i = 1
    j = nJ - 2
    dirichlet_function(i, j, boundaries=[1, 2])

    i = nI - 2
    j = nJ - 2
    dirichlet_function(i, j, boundaries=[2, 0])

    i = nI - 2
    j = 1
    dirichlet_function(i, j, boundaries=[3, 0])

    # Solve for T using Gauss-Seidel
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            T[i, j] = (T[i + 1, j] * coeffsT[i, j, 0] + T[i - 1, j] * coeffsT[i, j, 1] + T[i, j + 1] * coeffsT[
                i, j, 2] + T[i, j - 1] * coeffsT[i, j, 3] + S_U[i, j]) / coeffsT[i, j, 4]
    # Copy T to boundaries where homogeneous Neumann needs to be applied
    T[0, :] = T[1, :]
    # Compute residuals (taking into account normalization)
    temp_r_sum = 0
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            temp_r_sum += abs(coeffsT[i, j, 4] * T[i, j] - (
                        coeffsT[i, j, 0] * T[i + 1, j] + coeffsT[i, j, 1] * T[i - 1, j] + coeffsT[i, j, 2] * T[
                    i, j + 1] + coeffsT[i, j, 3] * T[i, j - 1] + S_U[i, j]))
    F = 0
    [dT_dx, dT_dy] = np.gradient(T, xCoords_N[:, 0], yCoords_N[0, :])

    for i in range(nI):
        F += abs(k[i, 0] * dx_CV[i, 1] * dT_dy[i, 0])
        F += abs(k[i, -1] * dx_CV[i, -2] * dT_dy[i, -1])

    for j in range(nJ):
        F += abs(k[0, j] * dy_CV[1, j] * dT_dx[0, j])
        F += abs(k[-1, j] * dy_CV[-2, j] * dT_dx[-1, j])
    r = temp_r_sum / F
    x.append(iter)
    residuals.append(r)

    print('iteration: %d\nresT = %.5e\n\n' % (iter, residuals[-1]))

    #  Check convergence
    if resTolerance > residuals[-1]:
        break
#print('iteration: %d\nresT = %.5e\n\n' % (iter, residuals[-1]))
# Compute heat fluxes
[dT_dx, dT_dy] = np.gradient(T, xCoords_N[:, 0], yCoords_N[0, :])

# Plotting section (these are some examples, more plots might be needed)

# Larger mesh plot
plt.plot(xCoords_M, yCoords_M)
plt.plot(np.transpose(xCoords_M), np.transpose(yCoords_M))
plt.title(str(mI) + " by " + str(mJ) + " " + str(grid_type) + " mesh ")

# larger heat flux plot
plt.figure()
plt.quiver(xCoords_N, yCoords_N, dT_dx, dT_dy)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Heat flux')

# larger temperature plot
plt.figure()
plt.contourf(xCoords_N, yCoords_N, T)
plt.colorbar()
plt.title('Temperature')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# residual plot
plt.figure()
plt.semilogy(residuals)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
plt.title('Residual')

# Subplots
plt.figure()
plt.suptitle(str(mI) + " by " + str(mJ) + " " + str(grid_type) + " mesh ")
# Plot mesh
plt.subplot(1, 2, 1)
plt.quiver(xCoords_N, yCoords_N, dT_dx, dT_dy)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Heat flux')

# Plot temperature contour
plt.subplot(1, 2, 2)
plt.contourf(xCoords_N, yCoords_N, T)
plt.colorbar()
plt.title('Temperature')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Plot residual convergence
#plt.subplot(2, 2, 3)
#plt.semilogx(x, residuals)
#plt.title('Residual convergence')
#plt.xlabel('iterations')
#plt.ylabel('residuals [-]')
#plt.title('Residual')

# Plot heat fluxes
#plt.subplot(2, 2, 4)
#plt.quiver(xCoords_N, yCoords_N, dT_dx, dT_dy)
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.title('Heat flux')
#plt.axis('equal')

plt.show()
