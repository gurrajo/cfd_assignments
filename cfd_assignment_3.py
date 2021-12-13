# MTF072 Computational Fluid Dynamics
# Task 3: laminar lid-driven cavity
# Template prepared by:
# Gonzalo Montero Villar
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# December 2020
# ==============Packages needed=================
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

# ================= Inputs =====================
# Fluid properties and B. C. inputs
UWall =  1# velocity of the upper wall
rho =  1# density
nu =  1/1000# kinematic viscosity
data_file = "data_Hybrid.txt" # data file where the given solution is stored
# Geometric inputs (fixed so that a fair comparison can be made)
mI = 11  # number of mesh points X direction.
mJ = 11  # number of mesh points Y direction.
xL = 1  # length in X direction
yL = 1  # length in Y direction
# Solver inputs
nIterations = 100 # maximum number of iterations
n_inner_iterations_gs =  10# amount of inner iterations when solving
# pressure correction with Gauss-Seidel
resTolerance =  0.001# convergence criteria for residuals
# each variable
alphaUV = 0.7 # under relaxation factor for U and V
alphaP = 0.3 # under relaxation factor for P
# ================ Code =======================
# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate, (i+1) is east and (j+1) north
# Allocate all needed variables
nI = mI + 1  # number of nodes in the X direction. nodes
# added in the boundaries
nJ = mJ + 1  # number of nodes in the Y direction. nodes
# added in the boundaries
coeffsUV = np.zeros((nI, nJ, 5))  # coefficients for the U and V equation
# E, W, N, S and P
sourceUV = np.zeros((nI, nJ, 2))  # source coefficients for the U and V equation
# U and V
coeffsPp = np.zeros((nI, nJ, 5))  # coefficients for the pressure correction
# equation E, W, N, S and P
sourcePp = np.zeros((nI, nJ))  # source coefficients for the pressure
# correction equation
U = np.zeros((nI, nJ))  # U velocity matrix
V = np.zeros((nI, nJ))  # V velocity matrix
P = np.zeros((nI, nJ))  # pressure matrix
Pp = np.zeros((nI, nJ))  # pressure correction matrix
massFlows = np.zeros((nI, nJ, 4))  # mass flows at the faces
# m_e, m_w, m_n and m_s
residuals = np.zeros((3, 1))  # U, V and conitnuity residuals
# Generate mesh and compute geometric variables
# Allocate all variables matrices
xCoords_N = np.zeros((nI, nJ))  # X coords of the nodes
yCoords_N = np.zeros((nI, nJ))  # Y coords of the nodes
xCoords_M = np.zeros((mI, mJ))  # X coords of the mesh points
yCoords_M = np.zeros((mI, mJ))  # Y coords of the mesh points
dWE = np.zeros((nI,nJ))
dSN = np.zeros((nI,nJ))
dxe_N = np.zeros((nI, nJ))  # X distance to east node
dxw_N = np.zeros((nI, nJ))  # X distance to west node
dyn_N = np.zeros((nI, nJ))  # Y distance to north node
dys_N = np.zeros((nI, nJ))  # Y distance to south node
dx_CV = np.zeros((nI, nJ))  # X size of the node
dy_CV = np.zeros((nI, nJ))  # Y size of the node
D = np.zeros((nI, nJ, 4))  # diffusive coefficients e, w, n and s
F = np.zeros((nI, nJ, 4))  # convective coefficients e, w, n and s
residuals_U = []
residuals_V = []
residuals_c = []
dx = xL / (mI - 1)
dy = yL / (mJ - 1)
# Fill the coordinates
for i in range(mI):
    for j in range(mJ):
        # For the mesh points
        xCoords_M[i, j] = i * dx
        yCoords_M[i, j] = j * dy
        # For the nodes
        if i > 0:
            xCoords_N[i, j] = 0.5 * (xCoords_M[i, j] + xCoords_M[i - 1, j])
        if i == mI - 1 and j > 0:
            yCoords_N[i + 1, j] = 0.5 * (yCoords_M[i, j] + yCoords_M[i, j - 1])
        if j > 0:
            yCoords_N[i, j] = 0.5 * (yCoords_M[i, j] + yCoords_M[i, j - 1])
        if j == mJ - 1 and i > 0:
            xCoords_N[i, j + 1] = 0.5 * (xCoords_M[i, j] + xCoords_M[i - 1, j])
        # Fill dx_CV and dy_CV
        if i > 0:
            dx_CV[i, j] = xCoords_M[i, j] - xCoords_M[i - 1, j]
        if j > 0:
            dy_CV[i, j] = yCoords_M[i, j] - yCoords_M[i, j - 1]
xCoords_N[-1, :] = xL
yCoords_N[:, -1] = yL
# Fill dxe, dxw, dyn and dys
for i in range(1, nI - 1):
    for j in range(1, nJ - 1):
        dxe_N[i, j] = xCoords_N[i + 1, j] - xCoords_N[i, j]
        dxw_N[i, j] = xCoords_N[i, j] - xCoords_N[i - 1, j]
        dyn_N[i, j] = yCoords_N[i, j + 1] - yCoords_N[i, j]
        dys_N[i, j] = yCoords_N[i, j] - yCoords_N[i, j - 1]
for i in range(2,nI-2):
    for j in range(2,nJ-2):
        dWE[i,j] = dx_CV[i,j] + dx_CV[i-1,j]/2 + dx_CV[i+1,j]/2
        dSN[i,j] = dy_CV[i,j] + dy_CV[i,j-1]/2 + dy_CV[i,j+1]/2
# Initialize variable matrices


# Looping
for iter in range(nIterations):
    # Impose boundary conditions for velocities, only the top boundary wall
    # is moving from left to right with UWall
    U[:, nJ - 1] = UWall
    # Impose pressure boundary condition, all homogeneous Neumann

    # Compute coefficients for U and V equations
    ## Compute coefficients for nodes one step inside the domain
    # diffusive coefficiants
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            D[i, j, 0] = dy_CV[i,j] * nu / dxe_N[i,j]  # east diffusive
            D[i, j, 1] = dy_CV[i,j] * nu / dxw_N[i,j]  # west diffusive
            D[i, j, 2] = dx_CV[i,j] * nu / dyn_N[i,j]  # north diffusive
            D[i, j, 3] = dx_CV[i,j] * nu / dys_N[i,j]  # south diffusive

    # convective coefficiants
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            fxe = 0.5 * dx_CV[i,j] / dxe_N[i,j]
            fxw = 0.5 * dx_CV[i,j] / dxw_N[i,j]
            fyn = 0.5 * dy_CV[i,j] / dyn_N[i,j]
            fys = 0.5 * dy_CV[i,j] / dys_N[i,j]

            Ue = fxe * U[i + 1, j] + (1 - fxe) * U[i, j]
            Uw = fxw * U[i - 1, j] + (1 - fxw) * U[i, j]
            Vn = fyn * V[i, j + 1] + (1 - fyn) * V[i, j]
            Vs = fys * V[i, j - 1] + (1 - fys) * V[i, j]

            F[i, j, 0] = dy_CV[i,j] * rho * Ue  # east convective
            F[i, j, 1] = dy_CV[i,j] * rho * Uw  # west convective
            F[i, j, 2] = dx_CV[i,j] * rho * Vn  # north convective
            F[i, j, 3] = dx_CV[i,j] * rho * Vs  # south convective


    ### First, north and south boundaries
    for i in range(2, nI - 2):
        j = 1
        coeffsUV[i, j, 0] = D[i,j,0] + np.max((0,-F[i,j,0]))
        coeffsUV[i, j, 1] = D[i,j,1] + np.max((0,F[i,j,1]))
        coeffsUV[i, j, 2] = D[i,j,2] + np.max((0,-F[i,j,2]))
        coeffsUV[i, j, 3] = 0
        del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
        sourceUV[i, j, 0] += np.max((del_f, 0)) * U[i, j]
        sourceUV[i, j, 1] += np.max((del_f, 0)) * V[i, j]
        del_f = -np.max((-del_f, 0))
        coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
        sourceUV[i, j, 0] += -(P[i+1,j] - P[i-1,j])/(dx_CV[i,j] + dx_CV[i-1,j]/2 + dx_CV[i+1,j]/2)*dx_CV[i,j]*dy_CV[i,j]
        sourceUV[i, j, 1] += 0 # homogenous neumann for pressure
        sourceUV[i,j,0] += coeffsUV[i,j,4]*(1-alphaUV)*U[i,j]
        sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

        j = nJ-2
        coeffsUV[i, j, 0] = D[i, j, 0] + np.max((0, -F[i, j, 0]))
        coeffsUV[i, j, 1] = D[i, j, 1] + np.max((0, F[i, j, 1]))
        coeffsUV[i, j, 2] = 0
        coeffsUV[i, j, 3] = D[i,j,3] + np.max((0,F[i,j,3]))
        del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
        sourceUV[i, j, 0] += np.max((del_f, 0)) * U[i, j]
        sourceUV[i, j, 1] += np.max((del_f, 0)) * V[i, j]
        del_f = -np.max(-del_f, 0)
        coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
        sourceUV[i, j, 0] = -(P[i+1,j] - P[i-1,j])/(dx_CV[i,j] + dx_CV[i-1,j]/2 + dx_CV[i+1,j]/2)*dx_CV[i,j]*dy_CV[i,j]
        sourceUV[i, j, 1] = 0
        sourceUV[i,j,0] += coeffsUV[i,j,4]*(1-alphaUV)*U[i,j]
        sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    ### Second, east and west boundaries
    for j in range(2, nJ - 2):
        i = 1
        coeffsUV[i, j, 0] = D[i, j, 0] + np.max((0, -F[i, j, 0]))
        coeffsUV[i, j, 1] = 0
        coeffsUV[i, j, 2] = D[i, j, 2] + np.max((0, -F[i, j, 2]))
        coeffsUV[i, j, 3] = D[i,j,3] + np.max((0,F[i,j,3]))
        del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
        sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
        sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
        del_f = -np.max(-del_f, 0)
        coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
        sourceUV[i, j, 0] = 0 # homogenous neumann for pressure
        sourceUV[i, j, 1] = -(P[i, j+1] - P[i, j-1]) / (dy_CV[i, j] + dy_CV[i - 1, j] / 2 + dy_CV[i + 1, j] / 2) * dx_CV[i, j] * dy_CV[i, j]
        sourceUV[i,j,0] += coeffsUV[i,j,4]*(1-alphaUV)*U[i,j]
        sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

        i = nJ - 2
        coeffsUV[i, j, 0] = 0
        coeffsUV[i, j, 1] = D[i, j, 1] + np.max((0, F[i, j, 1]))
        coeffsUV[i, j, 2] = D[i, j, 2] + np.max((0, -F[i, j, 2]))
        coeffsUV[i, j, 3] = D[i, j, 3] + np.max((0, F[i, j, 3]))
        del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
        sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
        sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
        del_f = -np.max(-del_f, 0)
        coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
        sourceUV[i, j, 0] = 0  # homogenous neumann for pressure
        sourceUV[i, j, 1] = -(P[i, j + 1] - P[i, j - 1]) / (dy_CV[i, j] + dy_CV[i - 1, j] / 2 + dy_CV[i + 1, j] / 2) * \
                            dx_CV[i, j] * dy_CV[i, j]
        sourceUV[i,j,0] += coeffsUV[i,j,4]*(1-alphaUV)*U[i,j]
        sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    ### Compute coefficients at corner nodes (one step inside)
    i = 1
    j = 1
    coeffsUV[i, j, 0] = D[i, j, 0] + np.max(0, -F[i, j, 0])
    coeffsUV[i, j, 1] = 0
    coeffsUV[i, j, 2] = D[i, j, 2] + np.max(0, -F[i, j, 2])
    coeffsUV[i, j, 3] = 0
    del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
    sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
    sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
    del_f = -np.max(-del_f, 0)
    coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
    sourceUV[i, j, 0] = 0 # homogenous neumann for pressure
    sourceUV[i, j, 1] = 0 # homogenous neumann for pressure
    sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
    sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    i = 1
    j = nJ-2
    coeffsUV[i, j, 0] = D[i, j, 0] + np.max(0, -F[i, j, 0])
    coeffsUV[i, j, 1] = 0
    coeffsUV[i, j, 2] = 0
    coeffsUV[i, j, 3] = D[i, j, 3] + np.max(0, F[i, j, 3])
    del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
    sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
    sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
    del_f = -np.max(-del_f, 0)
    coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
    sourceUV[i, j, 0] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 1] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
    sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    i = nI-2
    j = nJ-2
    coeffsUV[i, j, 0] = 0
    coeffsUV[i, j, 1] = D[i, j, 1] + np.max(0, F[i, j, 1])
    coeffsUV[i, j, 2] = 0
    coeffsUV[i, j, 3] = D[i, j, 3] + np.max(0, F[i, j, 3])
    del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
    sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
    sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
    del_f = -np.max(-del_f, 0)
    coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
    sourceUV[i, j, 0] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 1] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
    sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    i = nI-2
    j = 1
    coeffsUV[i, j, 0] = 0
    coeffsUV[i, j, 1] = D[i, j, 1] + np.max(0, F[i, j, 1])
    coeffsUV[i, j, 2] = D[i, j, 2] + np.max(0, -F[i, j, 2])
    coeffsUV[i, j, 3] = 0
    del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
    sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
    sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
    del_f = -np.max(-del_f, 0)
    coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
    sourceUV[i, j, 0] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 1] = 0  # homogenous neumann for pressure
    sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
    sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]

    ## Compute coefficients for inner nodes
    for i in range(2, nI - 2):
        for j in range(2, nJ - 2):
            coeffsUV[i, j, 0] = D[i, j, 0] + np.max(0, -F[i, j, 0])
            coeffsUV[i, j, 1] = D[i, j, 1] + np.max(0, F[i, j, 1])
            coeffsUV[i, j, 2] = D[i, j, 2] + np.max(0, -F[i, j, 2])
            coeffsUV[i, j, 3] = D[i, j, 3] + np.max(0, F[i, j, 3])
            del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
            sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
            sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
            del_f = -np.max(-del_f, 0)
            coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, :]) - del_f)/alphaUV
            sourceUV[i, j, 0] = -(P[i+1,j] - P[i-1,j])/(dx_CV[i,j] + dx_CV[i-1,j]/2 + dx_CV[i+1,j]/2)*dx_CV[i,j]*dy_CV[i,j]
            sourceUV[i, j, 1] = -(P[i, j + 1] - P[i, j - 1]) / (dy_CV[i, j] + dy_CV[i - 1, j] / 2 + dy_CV[i + 1, j] / 2) * dx_CV[i, j] * dy_CV[i, j]
            sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
            sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]


    ## Solve for U and V using Gauss-Seidel
    for i in range(iter_gs):
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                U[i,j] = (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] +coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0])/coeffsUV[i,j,4]
                V[i,j] = (coeffsUV[i,j,0]*V[i+1,j] +coeffsUV[i,j,1]*V[i-1,j] +coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1]  + sourceUV[i,j,1])/coeffsUV[i,j,4]

        for i in range(nI-2,0,-1):
            for j in range(nJ-2,0,-1):
                U[i,j] = (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] +coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0])/coeffsUV[i,j,4]
                V[i,j] = (coeffsUV[i,j,0]*V[i+1,j] +coeffsUV[i,j,1]*V[i-1,j] +coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1]  + sourceUV[i,j,1])/coeffsUV[i,j,4]

    ## Calculate at the faces using Rhie-Chow for the face velocities
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            fxe = 0.5 * dx_CV[i,j] / dxe_N[i,j]
            fxw = 0.5 * dx_CV[i,j] / dxw_N[i,j]
            fyn = 0.5 * dy_CV[i,j] / dyn_N[i,j]
            fys = 0.5 * dy_CV[i,j] / dys_N[i,j]

            Ue = fxe * U[i + 1, j] + (1 - fxe) * U[i, j]
            Uw = fxw * U[i - 1, j] + (1 - fxw) * U[i, j]
            Vn = fyn * V[i, j + 1] + (1 - fyn) * V[i, j]
            Vs = fys * V[i, j - 1] + (1 - fys) * V[i, j]

            F[i, j, 0] = dy_CV[j] * rho * Ue  # east convective
            F[i, j, 1] = dy_CV[j] * rho * Uw  # west convective
            F[i, j, 2] = dx_CV[i] * rho * Vn  # north convective
            F[i, j, 3] = dx_CV[i] * rho * Vs  # south convective

    for i in range(2,nI-2):
        for j in range(2,nJ-2):

            ue = (U[i,j] + U[i+1,j])/2 + dy_CV[i,j]/(4*coeffsUV[i,j,4])*(P[i+2,j] - 3*P[i+1,j] + 3*P[i,j] - P[i-1,j])

            vn = (V[i,j] + V[i,j+1])/2 + dx_CV[i,j]/(4*coeffsUV[i,j,4])*(P[i,j+2] - 3*P[i,j+1] + 3*P[i,j] - P[i,j-1])

            F[i,j,0] = dy_CV[j] * rho * ue
            F[i,j,2] = dx_CV[i] * rho * vn
            F[i,j+1,3] = F[i,j,2]
            F[i+1,j,1] = F[i,j,1]
        F[i+1,j,1] = F[i,j,0]
        F[i,j+1,3] = F[i,j,2]


    ## Calculate pressure correction equation coefficients
    D = np.zeros((nI, nJ, 4))
    for i in range(2,nI-2):
        for j in range(2,nJ-2):
            D[i,j,0] = 2/(coeffsUV[i,j,4] + coeffsUV[i+1,j,4])
            D[i, j, 1] = 2/(coeffsUV[i,j,4] + coeffsUV[i-1,j,4])
            D[i, j, 2] = 2/(coeffsUV[i,j,4] + coeffsUV[i,j+1,4])
            D[i, j, 3] = 2/(coeffsUV[i,j,4] + coeffsUV[i,j-1,4])

    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            # hint: set homogeneous Neumann coefficients with if

            coeffsPp[i, j, 0] = rho*dy_CV[i,j]*D[i,j,0]

            coeffsPp[i, j, 1] = rho*dy_CV[i,j]*D[i,j,1]

            coeffsPp[i, j, 2] = rho*dx_CV[i,j]*D[i,j,2]

            coeffsPp[i, j, 3] = rho*dx_CV[i,j]*D[i,j,3]

            coeffsPp[i, j, 4] = np.sum(coeffsPp[i,j,:])
            sourcePp[i, j] = F[i,j,1]-F[i,j,0] + F[i,j,3] - F[i,j,2]



    # Solve for pressure correction (Note that more that one loop is used)

    for iter_gs in range(n_inner_iterations_gs):
        for j in range(1, nJ - 1):
            for i in range(1, nI - 1):
                Pp[i,j] = (coeffsPp[i,j,0]*Pp[i+1,j] + coeffsPp[i,j,1]*Pp[i-1,j] + coeffsPp[i,j,2]*Pp[i,j+1] + coeffsPp[i,j,3]*Pp[i,j-1] + sourcePp[i,j])/coeffsPp[i,j,4]
        for i in range(nI - 1,1,-1):
            for j in range(nJ - 1,1,-1):
                Pp[i, j] = (coeffsPp[i, j, 0] * Pp[i + 1, j] + coeffsPp[i, j, 1] * Pp[i - 1, j] + coeffsPp[i, j, 2] *
                            Pp[i, j + 1] + coeffsPp[i, j, 3] * Pp[i, j - 1] + sourcePp[i, j]) / coeffsPp[i, j, 4]

        Pp -= Pp[2,2]# Set Pp with reference to node (2,2) and copy to boundaries
    Pp[:,0] = Pp[:,1]
    Pp[:,nJ-1] = Pp[:,nJ-2]
    Pp[0, :] = Pp[0, :]
    Pp[nJ-1, :] = Pp[nJ-2, :]
    # Correct velocities, pressure and mass flows
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            P[i,j] += alphaP*Pp[i,j]
            U[i,j] += ((Pp[i-1,j] + Pp[i,j])/2 - (Pp[i,j] + Pp[i+1,j])/2)/coeffsUV[i,j,4]
            V[i,j] += ((Pp[i,j-1] + Pp[i,j])/2 - (Pp[i,j] + Pp[i,j+1])/2)/coeffsUV[i,j,4]

    # impose zero mass flow at the boundaries
    U[0,:] = 0
    U[nI-1,:] = 0
    V[:,0] = 0
    V[:,nJ-1] = 0
    # Copy P to boundaries
    P[:, 0] = P[:, 1]
    P[:, nJ - 1] = P[:, nJ - 2]
    P[0, :] = P[0, :]
    P[nJ - 1, :] = P[nJ - 2, :]
    # Compute residuals
    residuals_U.append(0)  # U momentum residual
    residuals_V.append(0)  # V momentum residual
    residuals_c.append(0)  # continuity residual
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            residuals_U[-1] += abs(coeffsUV[i,j,4]*U[i,j] - (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] + coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + coeffsUV[i,j,0]))
            residuals_V[-1] += abs(coeffsUV[i,j,4]*U[i,j] - (coeffsUV[i,j,0]*V[i+1,j] + coeffsUV[i,j,1]*V[i-1,j] + coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1] + coeffsUV[i,j,1]))
            residuals_c[-1] += sourcePp[i,j]
    print('iteration: %d\nresU = %.5e, resV = %.5e, resCon = %.5e\n\n' \
          % (iter, residuals_U[-1], residuals_V[-1], residuals_c[-1]))

    #  Check convergence
    if resTolerance > max([residuals_U[-1], residuals_V[-1], residuals_c[-1]]):
        break
# Plotting section (these are some examples, more plots might be needed)
# Plot mesh
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Computational mesh')
# Plot results
plt.figure()
# U velocity contour
plt.subplot(2, 3, 1)
plt.title('U velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# V velocity contour
plt.subplot(2, 3, 2)
plt.title('V velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# P contour
plt.subplot(2, 3, 3)
plt.title('Pressure [Pa]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# Vector plot
plt.subplot(2, 3, 4)
plt.title('Vector plot of the velocity field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# Comparison with data
data = np.genfromtxt(data_file, skip_header=1)
uInterp = np.zeros((nJ - 2, 1))
vInterp = np.zeros((nJ - 2, 1))
for j in range(1, nJ - 1):
    for i in range(1, nI - 1):
        if xCoords_N[i, j] < 0.5 and xCoords_N[i + 1, j] > 0.5:
            uInterp[j - 1] = (U[i + 1, j] + U[i, j]) * 0.5
            vInterp[j - 1] = (V[i + 1, j] + V[i, j]) * 0.5
            break
        elif abs(xCoords_N[i, j] - 0.5) < 0.000001:
            uInterp[j - 1] = U[i, j]
            vInterp[j - 1] = V[i, j]
            break
plt.subplot(2, 3, 5)
plt.plot(data[:, 0], data[:, 2], 'r.', markersize=20, label='data U')
plt.plot(data[:, 1], data[:, 2], 'b.', markersize=20, label='data V')
plt.plot(uInterp, yCoords_N[1, 1:-1], 'k', label='sol U')
plt.plot(vInterp, yCoords_N[1, 1:-1], 'g', label='sol V')
plt.title('Comparison with data at x = 0.5')
plt.xlabel('u, v [m/s]')
plt.ylabel('y [m]')
plt.legend()
plt.subplot(2, 3, 6)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
# plt.legend('U momentum','V momentum', 'Continuity')
plt.title('Residuals')
plt.show()