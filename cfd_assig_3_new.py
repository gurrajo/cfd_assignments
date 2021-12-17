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
UWall = 1 # velocity of the upper wall
rho = 1 # density
nu = 1/1000# kinematic viscosity
mu = nu*rho
data_file = "data_Hybrid.txt" # data file where the given solution is stored
# Geometric inputs (fixed so that a fair comparison can be made)
mI = 11  # number of mesh points X direction.
mJ = 11  # number of mesh points Y direction.
xL = 1  # length in X direction
yL = 1  # length in Y direction
# Solver inputs
nIterations = 300 # maximum number of iterations
n_inner_iterations_gs = 20# amount of inner iterations when solving
# pressure correction with Gauss-Seidel
resTolerance =  0.001# convergence criteria for residuals
# each variable
alphaUV = 0.09 # under relaxation factor for U and V
alphaP = 0.5 # under relaxation factor for P
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
massFlows = np.zeros((nI, nJ, 4))  # mass flows at the faces
# m_e, m_w, m_n and m_s
  # U, V and conitnuity residuals
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
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        D[i, j, 0] = dy_CV[i,j] * mu / dxe_N[i,j]  # east diffusive
        D[i, j, 1] = dy_CV[i,j] * mu / dxw_N[i,j]  # west diffusive
        D[i, j, 2] = dx_CV[i,j] * mu / dyn_N[i,j]  # north diffusive
        D[i, j, 3] = dx_CV[i,j] * mu / dys_N[i,j]  # south diffusive
U[:, nJ - 1] = UWall
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
# Looping
for iter in range(nIterations):
    # Impose boundary conditions for velocities, only the top boundary wall
    # is moving from left to right with UWall
    U[:, nJ - 1] = UWall
    # Impose pressure boundary condition, all homogeneous Neumann

    # Compute coefficients for U and V equations
    ## Compute coefficients for nodes one step inside the domain


    ### First, north and south boundaries
    sourceUV = np.zeros((nI, nJ, 2))
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):

            coeffsUV[i, j, 0] = D[i, j, 0] + np.max((0, -F[i, j, 0]))
            coeffsUV[i, j, 1] = D[i, j, 1] + np.max((0, F[i, j, 1]))
            coeffsUV[i, j, 2] = D[i, j, 2] + np.max((0, -F[i, j, 2]))
            coeffsUV[i, j, 3] = D[i, j, 3] + np.max((0, F[i, j, 3]))
            del_f = F[i, j, 1] - F[i, j, 0] + F[i, j, 3] - F[i, j, 2]
            sourceUV[i, j, 0] += np.max(del_f, 0) * U[i, j]
            sourceUV[i, j, 1] += np.max(del_f, 0) * V[i, j]
            del_f = -np.max((-del_f, 0))
            coeffsUV[i, j, 4] = (np.sum(coeffsUV[i, j, 0:4]) - del_f)/alphaUV
            sourceUV[i, j, 0] += -(P[i+1,j] - P[i-1,j])/(dx_CV[i,j] + dx_CV[i-1,j]/2 + dx_CV[i+1,j]/2)*dx_CV[i,j]*dy_CV[i,j]
            sourceUV[i, j, 1] += -(P[i, j + 1] - P[i, j - 1]) / (dy_CV[i, j] + dy_CV[i - 1, j] / 2 + dy_CV[i + 1, j] / 2) * dx_CV[i, j] * dy_CV[i, j]
            sourceUV[i, j, 0] += coeffsUV[i, j, 4] * (1 - alphaUV) * U[i, j]
            sourceUV[i, j, 1] += coeffsUV[i, j, 4] * (1 - alphaUV) * V[i, j]



    ## Solve for U and V using Gauss-Seidel
    for iter_gs in range(n_inner_iterations_gs):
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                U[i,j] = (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] +coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0])/coeffsUV[i,j,4]
                V[i,j] = (coeffsUV[i,j,0]*V[i+1,j] + coeffsUV[i,j,1]*V[i-1,j] +coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1] + sourceUV[i,j,1])/coeffsUV[i,j,4]

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

            F[i, j, 0] = dy_CV[i,j] * rho * Ue  # east convective
            F[i, j, 1] = dy_CV[i,j] * rho * Uw  # west convective
            F[i, j, 2] = dx_CV[i,j] * rho * Vn  # north convective
            F[i, j, 3] = dx_CV[i,j] * rho * Vs  # south convective

    for i in range(1,nI-2):
        for j in range(1,nJ-2):
            # equidistant mesh for internal nodes
            ue = (U[i, j] + U[i + 1, j]) / 2 + dy_CV[i, j] / (4 * coeffsUV[i, j, 4]) * (P[i + 2, j] - 3 * P[i + 1, j] + 3 * P[i, j] - P[i - 1, j])

            vn = (V[i, j] + V[i, j + 1]) / 2 + dx_CV[i, j] / (4 * coeffsUV[i, j, 4]) * (P[i, j + 2] - 3 * P[i, j + 1] + 3 * P[i, j] - P[i, j - 1])

            if i == 1:
                # half the distance between P and west
                ue = (U[i,j] + U[i+1,j])/2 + dy_CV[i,j]/(4*coeffsUV[i,j,4])*(P[i+2,j] - 3*P[i+1,j] + 6*P[i,j] - 2*P[i-1,j])
            elif j == 1:
                # half the distance between P and south
                vn = (V[i, j] + V[i, j + 1]) / 2 + dx_CV[i, j] / (4 * coeffsUV[i, j, 4]) * (P[i, j + 2] - 3 * P[i, j + 1] + 6 * P[i, j] - 2*P[i, j - 1])

            F[i,j,0] = dy_CV[i,j] * rho * ue
            F[i,j,2] = dx_CV[i,i] * rho * vn
            F[i,j+1,3] = F[i,j,2]
            F[i+1,j,1] = F[i,j,0]

    ## Calculate pressure correction equation coefficients
    Dp = np.zeros((nI, nJ, 4))
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Dp[i,j,0] = 2/(coeffsUV[i,j,4] + coeffsUV[i+1,j,4])
            Dp[i, j, 1] = 2/(coeffsUV[i,j,4] + coeffsUV[i-1,j,4])
            Dp[i, j, 2] = 2/(coeffsUV[i,j,4] + coeffsUV[i,j+1,4])
            Dp[i, j, 3] = 2/(coeffsUV[i,j,4] + coeffsUV[i,j-1,4])

    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            # hint: set homogeneous Neumann coefficients with if

            coeffsPp[i, j, 0] = rho*dy_CV[i,j]*Dp[i,j,0]

            coeffsPp[i, j, 1] = rho*dy_CV[i,j]*Dp[i,j,1]

            coeffsPp[i, j, 2] = rho*dx_CV[i,j]*Dp[i,j,2]

            coeffsPp[i, j, 3] = rho*dx_CV[i,j]*Dp[i,j,3]
            if i == 1:
                coeffsPp[i,j,1] = 0
            if i == nI-2:
                coeffsPp[i,j,0] = 0
            if j == 1:
                coeffsPp[i,j,3] = 0
            if j == nJ-2:
                coeffsPp[i,j,2] = 0
            coeffsPp[i, j, 4] = np.sum(coeffsPp[i,j,0:4])
            sourcePp[i, j] = F[i,j,1]-F[i,j,0] + F[i,j,3] - F[i,j,2]

    coeffsPp[2, 2, 4] = -pow(10,30)
    # Solve for pressure correction (Note that more that one loop is used)
    Pp = np.zeros((nI, nJ))  # pressure correction matrix
    for iter_gs in range(n_inner_iterations_gs):
        for j in range(1, nJ - 1):
            for i in range(1, nI - 1):
                Pp[i,j] = (coeffsPp[i,j,0]*Pp[i+1,j] + coeffsPp[i,j,1]*Pp[i-1,j] + coeffsPp[i,j,2]*Pp[i,j+1] + coeffsPp[i,j,3]*Pp[i,j-1] + sourcePp[i,j])/coeffsPp[i,j,4]
        for i in range(nI - 2,0,-1):
            for j in range(nJ - 2,0,-1):
                Pp[i, j] = (coeffsPp[i, j, 0] * Pp[i + 1, j] + coeffsPp[i, j, 1] * Pp[i - 1, j] + coeffsPp[i, j, 2] *Pp[i, j + 1] + coeffsPp[i, j, 3] * Pp[i, j - 1] + sourcePp[i, j]) / coeffsPp[i, j, 4]

    Pp -= Pp[2,2] # Set Pp with reference to node (2,2) and copy to boundaries
    Pp[:,0] = Pp[:,1]
    Pp[:,nJ-1] = Pp[:,nJ-2]
    Pp[0, :] = Pp[1, :]
    Pp[nJ-1, :] = Pp[nJ-2, :]
    # Correct velocities, pressure and mass flows
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            P[i,j] += alphaP*Pp[i,j]
            U[i,j] += ((Pp[i-1,j] + Pp[i,j])/2 - (Pp[i,j] + Pp[i+1,j])/2)/coeffsUV[i,j,4]
            V[i,j] += ((Pp[i,j-1] + Pp[i,j])/2 - (Pp[i,j] + Pp[i,j+1])/2)/coeffsUV[i,j,4]

    for i in range(2,nI-2):
        for j in range(2,nJ-2):
            F[i, j, 0] += rho*Dp[i,j,0]*(Pp[i,j] - Pp[i+1,j])
            F[i, j, 1] += rho*Dp[i,j,1]*(Pp[i-1,j] - Pp[i,j])
            F[i, j, 2] += rho*Dp[i,j,2]*(Pp[i,j] - Pp[i,j+1])
            F[i, j, 3] += rho*Dp[i,j,3]*(Pp[i,j-1] - Pp[i,j])

    # impose zero mass flow at the boundaries
    F[nI-2,:,0] = 0
    F[1,:,1] = 0
    F[:,nJ-2,2] = 0
    F[:,1,3] = 0
    # Copy P to boundaries

    P[:,0] = P[:,1]
    P[:, nJ - 1] = P[:, nJ - 2]
    P[0, :] = P[1, :]
    P[nI - 1, :] = P[nI - 2, :]
    # Compute residuals
    residuals_U.append(0)  # U momentum residual
    residuals_V.append(0)  # V momentum residual
    residuals_c.append(0)  # continuity residual
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            residuals_U[-1] += abs(coeffsUV[i,j,4]*U[i,j] - (coeffsUV[i,j,0]*U[i+1,j] + coeffsUV[i,j,1]*U[i-1,j] + coeffsUV[i,j,2]*U[i,j+1] + coeffsUV[i,j,3]*U[i,j-1] + sourceUV[i,j,0]))
            residuals_V[-1] += abs(coeffsUV[i,j,4]*V[i,j] - (coeffsUV[i,j,0]*V[i+1,j] + coeffsUV[i,j,1]*V[i-1,j] + coeffsUV[i,j,2]*V[i,j+1] + coeffsUV[i,j,3]*V[i,j-1] + sourceUV[i,j,1]))
            residuals_c[-1] += sourcePp[i,j]
    print('iteration: %d\nresU = %.5e, resV = %.5e, resCon = %.5e\n\n' \
          % (iter, residuals_U[-1], residuals_V[-1], residuals_c[-1]))

    #  Check convergence
    if resTolerance > max([residuals_U[-1], residuals_V[-1], residuals_c[-1]]):
        break
# Plotting section (these are some examples, more plots might be needed)
# Plot mesh
plt.figure()
plt.plot(xCoords_M,yCoords_M)
plt.plot(xCoords_M.T,yCoords_M.T)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Computational mesh')
# Plot results
plt.figure()
# U velocity contour
plt.subplot(2, 3, 1)
plt.contourf(xCoords_N,yCoords_N,U)
plt.title('U velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# V velocity contour
plt.subplot(2, 3, 2)
plt.contourf(xCoords_N,yCoords_N,V)
plt.title('V velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# P contour
plt.subplot(2, 3, 3)
plt.contourf(xCoords_N,yCoords_N,P)
plt.title('Pressure [Pa]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
# Vector plot
plt.subplot(2, 3, 4)
plt.quiver(xCoords_N,yCoords_N, U, V)
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
residuals = np.zeros((iter+1, 2))
residuals[:,0] = residuals_U
residuals[:,1] = residuals_V
#residuals[:,2] = residuals_c
plt.semilogy(residuals)
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
plt.legend('UVC')
plt.title('Residuals')
plt.show()
