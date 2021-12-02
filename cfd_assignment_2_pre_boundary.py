# MTF072 Computational Fluid Dynamics
# Task 2: convection-diffusion equation
# Template prepared by:
# Gonzalo Montero Villar
# Department  of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# November 2019

# The script assumes that the folder with data is in the same path as this file

# Packages needed
import numpy as np
import matplotlib.pyplot as plt

def ReadDataAndGeometry(caseID, grid_type):
    if caseID <= 5:
        grid_number = 1
    elif caseID <= 10:
        grid_number = 2
    elif caseID <= 15:
        grid_number = 3
    elif caseID <= 20:
        grid_number = 4
    elif caseID <= 25:
        grid_number = 5

    path = 'data/grid%d/%s_grid' % (grid_number, grid_type)

    # Read data
    xCoords_M = np.genfromtxt('%s/xc.dat' % (path))  # x node coordinates
    yCoords_M = np.genfromtxt('%s/yc.dat' % (path))  # y node coordinates
    u_data = np.genfromtxt('%s/u.dat' % (path))  # U velocity at the nodes
    v_data = np.genfromtxt('%s/v.dat' % (path))  # V veloctiy at the nodes

    # Allocate geometrical data and variables
    mI = len(xCoords_M);  # number of mesh points in the x direction
    mJ = len(yCoords_M);  # number of mesh points in the y direction
    nI = mI + 1;  # number of nodes in the x direction
    nJ = mJ + 1;  # number of nodes in the y direction
    xCoords_N = np.zeros((nI, 1));  # mesh points x coordinates
    yCoords_N = np.zeros((nJ, 1));  # mesh points y coordinates
    dxe_N = np.zeros((nI, 1));  # X distance to east cell
    dxw_N = np.zeros((nI, 1));  # X distance to west cell
    dyn_N = np.zeros((nJ, 1));  # Y distance to north cell
    dys_N = np.zeros((nJ, 1));  # Y distance to south cell
    dx_CV = np.zeros((nI, 1));  # X size of the cell
    dy_CV = np.zeros((nJ, 1));  # Y size of the cell

    # Fill the cell coordinates as the mid point between mesh points, and at the same
    # position at the boundaries. Compute cell sizes
    for i in range(1, nI - 1):
        xCoords_N[i] = (xCoords_M[i] + xCoords_M[i - 1]) / 2
        dx_CV[i] = xCoords_M[i] - xCoords_M[i - 1]

    for j in range(1, nJ - 1):
        yCoords_N[j] = (yCoords_M[j] + yCoords_M[j - 1]) / 2
        dy_CV[j] = yCoords_M[j] - yCoords_M[j - 1]

    xCoords_N[0] = xCoords_M[0]
    xCoords_N[-1] = xCoords_M[-1]
    yCoords_N[0] = yCoords_M[0]
    yCoords_N[-1] = yCoords_M[-1]

    # Compute distances between nodes
    for i in range(1, nI - 1):
        dxe_N[i] = xCoords_N[i + 1] - xCoords_N[i]
        dxw_N[i] = xCoords_N[i] - xCoords_N[i - 1]

    for j in range(1, nJ - 1):
        dyn_N[j] = yCoords_N[j + 1] - yCoords_N[j]
        dys_N[j] = yCoords_N[j] - yCoords_N[j - 1]

    # Reshape the velocity data
    U = u_data.reshape(nI, nJ)
    V = v_data.reshape(nI, nJ)

    return [xCoords_M, yCoords_M, mI, mJ, nI, nJ, xCoords_N, yCoords_N, dxe_N, dxw_N, dyn_N, dys_N, dx_CV, dy_CV, U, V]


# Inputs

grid_type = 'fine'  # either 'coarse' or 'fine'
caseID = 20 # your case number to solve
k = 1
rho = 1 # density
nIterations = 10000 # number of iterations
Cp = 200
plotVelocityVectors = False
resTolerance = 0.001
TDMA = False
if TDMA:
    solver = "TDMA"
else:
    solver = "Gauss-Seidel"
# Read data for velocity fields and geometrical quantities

# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate, (i+1) is east and (j+1) north

[xCoords_M, yCoords_M, mI, mJ, nI, nJ, xCoords_N, yCoords_N, dxe_N, dxw_N, dyn_N, dys_N, dx_CV, dy_CV, U,
 V] = ReadDataAndGeometry(caseID, grid_type)

# Plot velocity vectors if required
if plotVelocityVectors:
    plt.figure()
    plt.quiver(xCoords_N, yCoords_N, U.T, V.T)
    plt.title('Velocity vectors')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

# Allocate needed vairables
T = np.ones((nI, nJ))*273.15  # temperature matrix
D = np.zeros((nI, nJ, 4))  # diffusive coefficients e, w, n and s
F = np.zeros((nI, nJ, 4))  # convective coefficients e, w, n and s
coeffsT = np.zeros((nI, nJ, 5))  # hybrid scheme coefficients E, W, N, S, P
S_U = np.zeros((nI, nJ))
S_P = np.zeros((nI, nJ))
TD = 263.15 #K

hai = np.array([1])
haj = np.array([45, 46, 47, 48])
hbi = np.array([1])
hbj = np.array([1, 2, 3, 4, 5, 6])
hci = np.array([48])
hcj = np.array([1, 2, 3, 4, 5, 6])
hdi = np.array([23, 24, 25, 26])
hdj = np.array([48])
dir_bound_i = np.array([1, 23, 24, 25, 26])
dir_bound_j = np.array(np.linspace(6,nJ-1,nJ-6))
q = 50
T[[0],45:50] = 263.15
T[hdi,[49]] = 263.15
T[[0],7:45] = 293.15

# rest is homogeneous neumann

residuals = []

# Code

gamma = k / Cp
## Diffusive and convective coefficient calculations
for i in range(1, nI - 1):
    for j in range(1, nJ - 1):
        D[i, j, 0] = gamma/dxe_N[i] # east diffusive
        D[i, j, 1] = gamma/dxw_N[i] # west diffusive
        D[i, j, 2] = gamma/dyn_N[j] # north diffusive
        D[i, j, 3] = gamma/dys_N[j] # south diffusive

        F[i, j, 0] = dy_CV[j]*rho*(U[i,j]+U[i+1,j])/2 # east convective
        F[i, j, 1] = dy_CV[j]*rho*(U[i,j]+U[i-1,j])/2 # west convective
        F[i, j, 2] = dx_CV[i]*rho*(V[i,j]+V[i,j+1])/2 # north convective
        F[i, j, 3] = dx_CV[i]*rho*(V[i,j]+V[i,j-1])/2 # south convective

# Hybrid scheme coefficients calculations (taking into account boundary conditions)

for i in range(2, nI - 2):
    y = [1,48]
    for j in y:
        coeffsT[i, j, 0] = np.max([-F[i, j, 0], 0, (D[i, j, 0] - F[i, j, 0] / 2)])
        coeffsT[i, j, 1] = np.max([F[i, j, 1], 0, (D[i, j, 1] + F[i, j, 1] / 2)])
        coeffsT[i, j, 2] = np.max([-F[i, j, 2], 0, (D[i, j, 2] - F[i, j, 2] / 2)])
        coeffsT[i, j, 3] = np.max([F[i, j, 3], 0, (D[i, j, 3] + F[i, j, 3] / 2)])

        if i in hdi and j in hdj:
            Tb = 263.15
            S_P[i,j] -= D[i,j,2] + F[i,j,2]
            S_U[i,j] += Tb*(D[i,j,2] + F[i,j,2])
            coeffsT[i,j,2] = 0
        elif j == 1:
            q = 50  # W/m^2
            S_U[i,j] = q*dx_CV[i]
            coeffsT[i,j,3] = 0
        else:
            # neumann
            coeffsT[i,j,2] = 0
        delF = F[i, j, 0] - F[i, j, 1] + F[i, j, 2] - F[i, j, 3]
        coeffsT[i, j, 4] = np.sum(coeffsT[i, j, 0:4]) + delF - S_P[i,j]

for j in range(2, nJ - 2):
    x = [1, 48]
    for i in x:
        coeffsT[i, j, 0] = np.max([-F[i, j, 0], 0, (D[i, j, 0] - F[i, j, 0] / 2)])
        coeffsT[i, j, 1] = np.max([F[i, j, 1], 0, (D[i, j, 1] + F[i, j, 1] / 2)])
        coeffsT[i, j, 2] = np.max([-F[i, j, 2], 0, (D[i, j, 2] - F[i, j, 2] / 2)])
        coeffsT[i, j, 3] = np.max([F[i, j, 3], 0, (D[i, j, 3] + F[i, j, 3] / 2)])
        if i in hai and j in haj:
            coeffsT[i, j, 1] = 0
            Tb = 263.15
            S_P[i, j] -= D[i, j, 1]
            S_U[i, j] += Tb*D[i, j, 1]
        elif i in hbi and j in hbj:
            # neumann
            coeffsT[i,j,1] = 0
        elif i in hci and j in hcj:
            # neumann
            coeffsT[i,j,0] = 0
        elif i == 1:
            Tb = 293.15
            S_P[i, j] -= D[i, j, 1]
            S_U[i, j] += Tb * D[i, j, 1]
            coeffsT[i,j,1] = 0
        else:
            if i == 1:
                coeffsT[i, j, 1] = 0
            else:
                coeffsT[i, j, 0] = 0
        delF = F[i, j, 0] - F[i, j, 1] + F[i, j, 2] - F[i, j, 3]
        coeffsT[i, j, 4] = np.sum(coeffsT[i, j, 0:4]) + delF - S_P[i,j]

# bot_left
i = 1
j = 1
q = 50  # W/m^2
coeffsT[i, j, 0] = np.max([-F[i,j,0] ,0, (D[i,j,0] - F[i,j,0]/2)])
coeffsT[i, j, 1] = 0
coeffsT[i, j, 2] = np.max([-F[i,j,2] ,0, (D[i,j,2] - F[i,j,2]/2)])
coeffsT[i, j, 3] = 0
delF = F[i,j,0] - F[i,j,1] + F[i,j,2] - F[i,j,3]
coeffsT[i, j, 4] = np.sum(coeffsT[i,j,0:4]) + delF
S_U[i,j] = q*dx_CV[i]

# top_left
i = 1
j = 48
coeffsT[i, j, 0] = np.max([-F[i,j,0] ,0, (D[i,j,0] - F[i,j,0]/2)])
coeffsT[i, j, 1] = 0
coeffsT[i, j, 2] = 0
coeffsT[i, j, 3] = np.max([F[i,j,3] ,0, (D[i,j,3] + F[i,j,3]/2)])
Tb = 263.15
S_P[i, j] -= D[i, j, 1]
S_U[i, j] += Tb * D[i, j, 1]
delF = F[i,j,0] - F[i,j,1] + F[i,j,2] - F[i,j,3]
coeffsT[i, j, 4] = np.sum(coeffsT[i,j,0:4]) + delF -S_P[i,j]

# top_right
i = 48
j = 48
coeffsT[i, j, 0] = 0
coeffsT[i, j, 1] = np.max([F[i,j,1] ,0, (D[i,j,1] + F[i,j,1]/2)])
coeffsT[i, j, 2] = 0
coeffsT[i, j, 3] = np.max([F[i, j, 3], 0, (D[i, j, 3] + F[i, j, 3] / 2)])
delF = F[i,j,0] - F[i,j,1] + F[i,j,2] - F[i,j,3]
coeffsT[i, j, 4] = np.sum(coeffsT[i,j,0:4]) + delF


# bot_right
i = 48
j = 1
coeffsT[i, j, 0] = 0
coeffsT[i, j, 1] = np.max([F[i,j,1] ,0, (D[i,j,1] + F[i,j,1]/2)])
coeffsT[i, j, 2] = np.max([-F[i,j,2] ,0, (D[i,j,2] - F[i,j,2]/2)])
coeffsT[i, j, 3] = 0
delF = F[i,j,0] - F[i,j,1] + F[i,j,2] - F[i,j,3]
coeffsT[i, j, 4] = np.sum(coeffsT[i,j,0:4]) + delF
S_U[i,j] = q*dx_CV[i]

for i in range(2, nI - 2):
    for j in range(2, nJ - 2):
        coeffsT[i, j, 0] = np.max([-F[i,j,0] ,0, (D[i,j,0] - F[i,j,0]/2)])
        coeffsT[i, j, 1] = np.max([F[i,j,1] ,0, (D[i,j,1] + F[i,j,1]/2)])
        coeffsT[i, j, 2] = np.max([-F[i,j,2] ,0, (D[i,j,2] - F[i,j,2]/2)])
        coeffsT[i, j, 3] = np.max([F[i,j,3] ,0, (D[i,j,3] + F[i,j,3]/2)])
        delF = F[i,j,0] - F[i,j,1] + F[i,j,2] - F[i,j,3]
        coeffsT[i, j, 4] = np.sum(coeffsT[i,j,0:4]) + delF

for iter in range(nIterations):
    # Impose boundary conditions

    # Solve for T using Gauss-Seidel or TDMA (both results need to be
    if TDMA:
        for j in range(1,nJ-1):
            P = np.zeros((nI,1))
            Q = np.zeros((nI, 1))
            for i in range(1,nI-1):
                d = coeffsT[i, j, 2] * T[i, j + 1] + coeffsT[i, j, 3] * T[i, j - 1] + S_U[i, j]
                if i == 1:
                    if i in dir_bound_i and j in dir_bound_j:
                        Tb = T[0,j]
                        P[i] = coeffsT[i,j,0]/coeffsT[i,j,4]
                        Q[i] = (d+coeffsT[i,j,1]*Tb)/coeffsT[i,j,4]
                    else:
                        # neumann
                        P[i] = coeffsT[i,j,0]/coeffsT[i,j,4]
                        Q[i] = d/coeffsT[i,j,4]
                elif i == nI - 2:
                    if i in dir_bound_i and j in dir_bound_j:
                        P[i] = 0
                        Q[i] = (d + coeffsT[i,j,1] * Q[i-1] + coeffsT[i,j,0] * T[i+1,j]) / (coeffsT[i,j,4] - coeffsT[i,j,1] * P[i-1])
                    else:
                        # neumann
                        P[i] = 0
                        Q[i] = (d + coeffsT[i,j,1] * Q[i-1])/ (coeffsT[i,j,4] - coeffsT[i,j,1] * P[i-1])
                else:
                    P[i] = coeffsT[i,j,0]/(coeffsT[i,j,4] - coeffsT[i,j,1]*P[i-1])
                    Q[i] = (d + coeffsT[i,j,1]*Q[i-1])/(coeffsT[i,j,4] - coeffsT[i,j,1]*P[i-1])
            for i in range(nI-2,0,-1):
                T[i,j] = P[i]*T[i+1,j] + Q[i]

        for i in range(1,nI-1):
            P = np.zeros((nJ,1))
            Q = np.zeros((nJ,1))
            for j in range(1,nJ-1):
                d = coeffsT[i, j, 0] * T[i + 1, j] + coeffsT[i, j, 1] * T[i - 1, j] + S_U[i, j]
                if j == 1:
                    if i in dir_bound_i and j in dir_bound_j:
                        Tb = T[i,0]
                        P[j] = coeffsT[i,j,2]/coeffsT[i,j,4]
                        Q[j] = (d+coeffsT[i,j,3]*Tb)/coeffsT[i,j,4]
                    else:
                        # neumann
                        P[j] = coeffsT[i,j,2]/coeffsT[i,j,4]
                        Q[j] = d/coeffsT[i,j,4]
                elif j == nJ - 2:
                    if i in dir_bound_i and j in dir_bound_j:
                        P[j] = 0
                        Q[j] = (d + coeffsT[i,j,3] * Q[j-1] + coeffsT[i,j,2] * T[i,j+1]) / (coeffsT[i,j,4] - coeffsT[i,j,3] * P[j-1])
                    else:
                        # neumann
                        P[j] = 0
                        Q[j] = (d + coeffsT[i,j,3] * Q[j-1])/(coeffsT[i,j,4] - coeffsT[i,j,3] * P[j-1])
                else:
                    P[j] = coeffsT[i,j,2]/(coeffsT[i,j,4] - coeffsT[i,j,3]*P[j-1])
                    Q[j] = (d + coeffsT[i,j,3]*Q[j-1])/(coeffsT[i,j,4] - coeffsT[i,j,3]*P[j-1])
            for j in range(nJ-2,0,-1):
                T[i,j] = P[j]*T[i,j+1] + Q[j]
    else:
        for i in range(1, nI - 1):
            for j in range(1, nJ - 1):
                T[i, j] = (T[i + 1, j] * coeffsT[i, j, 0] + T[i - 1, j] * coeffsT[i, j, 1] + T[i, j + 1] * coeffsT[i, j, 2] + T[i, j - 1] * coeffsT[i, j, 3] + S_U[i, j]) / coeffsT[i, j, 4]
    # Copy temperatures to boundaries
    T[0,0:7] = T[1,0:7]
    #T[0,46:50] = T[1,46:50]
    T[0:23,49] = T[0:23,48]
    T[27:50, 49] = T[27:50, 48]
    T[49,:] = T[48,:]
    T[:,0] = T[:,1] # needs to be looked at
    # Compute residuals (taking into account normalization)
    temp_sum_r = 0
    inlet_f = 0
    outlet_f = 0
    for i in range(1,nI-1):
        for j in range(i,nJ-1):
            temp_sum_r += abs(coeffsT[i, j, 4] * T[i, j] - (coeffsT[i, j, 0] * T[i + 1, j] + coeffsT[i, j, 1] * T[i - 1, j] + coeffsT[i, j, 2] * T[i, j + 1] + coeffsT[i, j, 3] * T[i, j - 1] + S_U[i, j]))
    for i in hdi:
        inlet_f += abs(rho*V[i,49]*dx_CV[i]*TD)
    for j in hbj:
        outlet_f -= abs(rho*U[0,j]*dy_CV[j]*T[0,j])
        outlet_f -= abs(rho*U[49,j]*dy_CV[j]*T[49,j])
    F = abs(inlet_f - outlet_f)
    r = temp_sum_r/F
    residuals.append(r)  # fill with your residual value for the
    # current iteration

    print('iteration: %d\nresT = %.5e\n\n' % (iter, residuals[-1]))

    # Check convergence

    if resTolerance > residuals[-1]:
        break
# check global conservation



# Plotting (these are some examples, more plots might be needed)
[dT_dx, dT_dy] = np.gradient(T, xCoords_N[:,0], yCoords_N[:,0])

global_con = 0

for i in hdi:
    global_con += abs(rho * V[i, 49] * dx_CV[i] * TD)
    global_con += gamma*dT_dy[i,48]
for j in range(7,50):
    global_con -= gamma*dT_dx[0,j]

for j in hbj:
    global_con -= abs(rho * U[0, j] * dy_CV[j] * T[0, j])
    global_con -= abs(rho * U[49, j] * dy_CV[j] * T[49, j])

# add source term
global_con += xCoords_M[-1]*q

print("global conservation: " + str(global_con))

plt.figure()
# dt_dx defined positive in the east direction and our normal vector is in the west direction, use -dt_dx
plt.quiver(np.ones((43,1)), yCoords_N[7:50], -dT_dx[1,7:50], dT_dy[1,7:50])
plt.title('Temperature gradient on dirichlet boundary')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
xv, yv = np.meshgrid(xCoords_N, yCoords_N)

plt.figure()
plt.quiver(xv, yv, U.T, V.T)
plt.title('Velocity vectors')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.figure()
plt.suptitle("grid type:" + str(grid_type) + " | Solver: " + solver)
plt.subplot(1, 2, 1)
plt.contourf(yv, xv, T)
plt.colorbar()
plt.title('Temperature')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.subplot(1, 2, 2)
plt.semilogy(residuals)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
plt.title('Residual')

plt.show()
