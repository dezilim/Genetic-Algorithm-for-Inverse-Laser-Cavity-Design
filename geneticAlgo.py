from fenics import *
from dolfin import *
import matplotlib.pyplot as plt

import numpy as np
from random import randrange

from mshr import *


import time
np.set_printoptions(threshold=np.inf)


try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

try:
    pyvista.set_jupyter_backend("pythreejs")
except ModuleNotFoundError:
    print("pythreejs is required for this demo")
    exit(0)

# import vtkRenderingOpenGL2
# from pyvista import _vtk
# from pyvista import *

# dimensions of box (kinda useless, I use N)
a = 50.0 
b = 50.0
c = 50.0

# set to 500 for a = 50 
nb_pts_a = 50
nb_pts_b = 50
nb_pts_c = 50


# taking into account the mirror thickness
mirror_w = 3

LAME_MU = 1.0
LAME_LAMBDA = 1.25
DENSITY = 1.0
ACCELERATION_DUE_TO_GRAVITY = 0.016



# ===========================
# Do not change the relative parameters of the above otherwise it will crumble and die
# ===========================

# in a 3d voxel grid we label from 0,1,2,3,... N*3-1
# here , N is the dimension 
def getCoordFromIndex(index, N):
    x =  index%N
    z = (index - (index%(N*N)))/(N*N)
    y = ((index- z*N*N) - (index- z*N*N)%N)/N
    return [x,y,z]

def getIndexFromCoord(xi, yi, zi, N):
    return (xi + yi*N + zi*N*N)


# REMOVE CUBES IN MESH USING COORDINATES / INDEX
# ===============================================
# takes the left bottom coordinates of the unit box to remove. (xi yi zi) < (a b c) 
def removeUnitCube(xi, yi, zi, this_box):
    box_to_remove = Box(Point(xi, yi, zi),Point(xi + 1, yi +1, zi +1))
    return (this_box - box_to_remove)

def removeUnitCubeFromIndex(index, N, this_box):
    coords = getCoordFromIndex(index, N)
#     print("Check removeUnitCubeFromIndex. Removing index %d with coordinates %f, %f, %f" % (index, coords[0], coords[1], coords[2]))
    this_box = removeUnitCube(coords[0], coords[1], coords[2], this_box)
    return (this_box)
    

# ===============================================

    
# MODIFY MESH USING COORDINATES 
    
# removes a unit vertical column. 
def removeVerticalCol(xi, yi, length, this_box):
    box_to_remove = Box(Point(xi, yi,0 + mirror_w), Point(xi+1, yi+1, length - mirror_w))

    return (this_box - box_to_remove)

def addVerticalCol(xi, yi, length, this_box):
    box_to_add = Box(Point(xi, yi,0 + mirror_w), Point(xi+1, yi+1, length - mirror_w))

    return (this_box + box_to_add)
# removes a vertical hole of an approximative circular shape from a mesh box
# xi and yi represent the center part of this shape 
#         * * *
#       * * * * *
#       * * & * *
#       * * * * *
#         * * * 


def removeVerticalHole(xi, yi, length, this_box):
    box_to_remove_center = Box(Point(xi-2, yi-1,0 + mirror_w), Point(xi+2, yi+1, length - mirror_w))
    box_to_remove_side1 = Box(Point(xi-1, yi-2,0 + mirror_w), Point(xi+1, yi-1, length - mirror_w))
    box_to_remove_side2 = Box(Point(xi-1, yi+1,0 + mirror_w), Point(xi+1, yi+2, length - mirror_w))
    
    return (this_box - box_to_remove_center - box_to_remove_side1 -box_to_remove_side2)

def baseVerticalHole(xi, yi, length):
    baseBox = Box(Point(xi - 4, yi -4, 0), Point(xi + 4, yi +4, length))
    # this will keep some parts for mirror width if needed
    baseHole = removeVerticalHole(xi, yi, length, baseBox)

    return (baseHole)

# ============================================================================
# MODIFY STATE VECTOR 
# ============================================================================

def psi_removeVerticalHole(N, psi_arr):
    # if even, remove a simple 4 block 
    if (N%2 == 0):
        # bottom left coord
        _ref = N/2 -1;
    else: 
        _ref = (N-1)/2
    print("Ref for psi_removeVerticalHole: ", _ref)
    # index without the z part. (x+1) + y*N 
    BL = _ref + _ref*N 
    BR = (_ref + 1) + _ref*N
    TL = _ref + (_ref+1)*N 
    TR = (_ref + 1) + (_ref+1)*N 

    # remove whole side chunk first 
    for x in np.arange(0,N):
        for y in np.arange(0,N):
            # z = 0
            psi_arr[int(x + y*N )] = 0
            # z = N-1
            psi_arr[int(x + y*N + (N-1)*N*N)] = 0
    # hole region
    for z in np.arange(0,N):
        if ((z == 0) or (z ==  N-1)):
            psi_arr[int(BL + z*N*N)] = 1
            psi_arr[int(BR + z*N*N)] = 1
            psi_arr[int(TL + z*N*N)] = 1
            psi_arr[int(TR + z*N*N)] = 1

            # add mirrors at side extra, x direction
            psi_arr[int(BL -1 + z*N*N)] = 1
            psi_arr[int(BR +1 + z*N*N)] = 1
            psi_arr[int(TL -1 + z*N*N)] = 1
            psi_arr[int(TR +1 + z*N*N)] = 1

            # add mirrors at side extra, y direction
            psi_arr[int(BL -N + z*N*N)] = 1
            psi_arr[int(BR -N + z*N*N)] = 1
            psi_arr[int(TL +N + z*N*N)] = 1
            psi_arr[int(TR +N + z*N*N)] = 1

            # add mirrors at side extra, corner direction
            psi_arr[int(BL -N -1 + z*N*N)] = 1
            psi_arr[int(BR -N +1 + z*N*N)] = 1
            psi_arr[int(TL +N -1+ z*N*N)] = 1
            psi_arr[int(TR +N +1+ z*N*N)] = 1

        else:   
            psi_arr[int(BL + z*N*N)] = 0
            psi_arr[int(BR + z*N*N)] = 0
            psi_arr[int(TL + z*N*N)] = 0
            psi_arr[int(TR + z*N*N)] = 0
            

#     # if odd, just remove the center 
#     else: 
#         _ref = (N-1)/2
#         print("Ref for psi_removeVerticalHole: ", _ref)
#         BL = _ref + _ref*N 
#         for z in np.arange(1, N-1):
#             psi_arr[int(BL + z*N*N)] = 0
            
    return (psi_arr)

def psi_removeHorizontalHole(N, psi_arr):
    # simplify it, if its odd just shift it slightly to the left 
    if (N%2 == 0):
        # bottom left coord
        _ref = N/2 -1
        
    else: 
        _ref = (N-1)/2
        
    print("Ref for psi_removeVerticalHole: ", _ref)
    # index without the z part. (x+1) + y*N 
    print("REF: BL, BR, TL, TR:") 
    BL = _ref + _ref*N*N 
    BR = (_ref + 1) + _ref*N*N
    TL = _ref + (_ref+1)*N*N 
    TR = (_ref + 1) + (_ref+1)*N*N 
    print(BL,  "\n",BR, "\n", TL, "\n", TR, "\n")


    # remove whole side chunk first 
    for x in np.arange(0,N):
        for z in np.arange(0,N):
            # y = 0
            psi_arr[int(x + z*N*N)] = 0
            # y = N-1
            psi_arr[int(x + (N-1)*N + z*N*N)] = 0
    # hole region
    for y in np.arange(0,N):
        if ((y == 0) or (y ==  N-1)):
            psi_arr[int(BL + y*N)] = 1
            psi_arr[int(BR + y*N)] = 1
            psi_arr[int(TL + y*N)] = 1
            psi_arr[int(TR + y*N)] = 1

            # add mirrors at side extra, x direction
            psi_arr[int(BL -1 +  y*N)] = 1
            psi_arr[int(BR +1 +  y*N)] = 1
            psi_arr[int(TL -1 +  y*N)] = 1
            psi_arr[int(TR +1 +  y*N)] = 1

            # add mirrors at side extra, z direction
            psi_arr[int(BL -N*N +  y*N)] = 1
            psi_arr[int(BR -N*N +  y*N)] = 1
            psi_arr[int(TL +N*N +  y*N)] = 1
            psi_arr[int(TR +N*N +  y*N)] = 1

            # add mirrors at side extra, corner direction
            psi_arr[int(BL -N*N -1 +  y*N)] = 1
            psi_arr[int(BR -N*N +1 +  y*N)] = 1
            psi_arr[int(TL +N*N -1 +  y*N)] = 1
            psi_arr[int(TR +N*N +1 +  y*N)] = 1

        else:   
            psi_arr[int(BL + y*N)] = 0
            psi_arr[int(BR + y*N)] = 0
            psi_arr[int(TL + y*N)] = 0
            psi_arr[int(TR + y*N)] = 0

                

            
            
#     # if odd, just remove the center 
#     else: 
#         _ref = (N-1)/2
#         print("Ref for psi_removeVerticalHole: ", _ref)
#         BL = _ref + _ref*N*N 
#         for y in np.arange(1, N-1):
#             print("Hole removal of index: ", int(BL + y*N))
#             psi_arr[int(BL + y*N)] = 0
            
    return (psi_arr)

def psi_removeCube(N, psi_arr, xi, yi, zi):
    index = getIndexFromCoord(xi, yi, zi, N)
    psi_arr[int(index)] = 0
    return (psi_arr)


# obviously, you cannot have xi yi or zi less than 0
def psi_removeChunk(N, psi_arr, xi, yi, zi, size):
    if ((xi + size > N) or (yi + size > N) or (zi + size > N)):
        raise Exception("Chunk to remove excceeds bounds")
        
    for _z in np.arange(zi, zi + size):
        for _y in np.arange(yi, yi + size):
            for _x in np.arange(xi, xi + size):
                psi_arr = psi_removeCube(N, psi_arr, _x,_y,_z)
    return (psi_arr)





# -------------------------------------------------------------------------------
# SIM FROM RANDOM CHUNKS HOLE
# -------------------------------------------------------------------------------

# Simulate a configuration with randomly chosen chunks with maximum sizes. Hole state can be horizontal or vertical. 
# Mesh res is for creating the mesh. Use 10 for N = 10
# maxSize_chunks should be less than the offset
def sim_randomChunks_hole(N, offset, nb_chunks, maxSize_chunks, hole_state, mesh_res, file_name):
    nb_voxels = N*N*N
    nb_empty = 0
    
    if (N%2 == 0):
        # bottom left coord
        _ref = N/2 -1
        
    else: 
        _ref = (N-1)/2
    psi = np.array([0] * nb_empty + [1] * (nb_voxels - nb_empty))
    
    
    # remove some chunks from the statevect
    # =====================================
    for i in range(nb_chunks):
        # get a random x,y,z coord and size to remove 
        xi = randrange(N-offset) ; yi = randrange(N-offset); zi = randrange(N-offset) +1; 
        size = randrange(maxSize_chunks) +1;
        print("Chosen (x,y,z, size): " , xi, yi, zi, size)
        psi = psi_removeChunk(N, psi, xi, yi, zi, size)
        print("=============================")

    # If none of these then dont remove hole
    if (hole_state == "HORIZONTAL"):
        psi = psi_removeHorizontalHole(N, psi)
    elif (hole_state == "VERTICAL"):
        psi = psi_removeVerticalHole(N, psi)
    
#     print("State Vector: ")
#     print(psi)
#     print("=============================")
    
    # INITIALIZE START BOX
    # ===================
    startBox = Box(Point(0.0, 0.0, 0.0), Point(N,N,N))
    thisMesh = startBox

    # MODIFY MESH WITH STATE VECTOR 
    # ===================
    for index in range(nb_voxels):
        if (psi[index] == 0):
            index_coord = getCoordFromIndex(index, N)
#             print("Removing index ", index, " with coords: ", index_coord[0], index_coord[1], index_coord[2])
            thisMesh = removeUnitCubeFromIndex(index, N, thisMesh)
            
            

    mesh = generate_mesh(thisMesh, mesh_res)
    
    # lagrange_vector_space_first_order
    V = VectorFunctionSpace(
        mesh,
        "Lagrange",
        1,
    )
    
    
    # testing the extracting node values.
    print("..................TEST......................................")
    dofmap = V.dofmap()
    dofs = dofmap.dofs()

    # Get coordinates of all global indices in len(nodes) x 3 array. In fact we are getting 3 repeated coordinates for each one. 
    dofs_x = V.tabulate_dof_coordinates() 
    # print("Dofs_x:\n", dofs_x)
    # print("SHAPE OF DOFS_x:" , len(dofs_x))

    # x component of all the nodes. ORIGINAL COORIDNATES. 
    x = dofs_x[::3,0]; y = dofs_x[::3,1]; z = dofs_x[::3,2]
    print("len of x (initial from dofs_x):", len(x))


    # MIRROR DOMAIN ===========================================
    offset = 0.05
    indices_left_mirror = np.where((y > 1-offset) & (y < 1+offset) & (z > _ref) & (z < _ref+2) & (x > _ref) & (x < _ref+2))[0]
    indices_right_mirror = np.where((y > N-1 -0.05) & (y < N -1+ 0.05) & (z > _ref) & (z < _ref+2) & (x > _ref) & (x < _ref+2))[0]

    # we focus on the y direction cos thats the optical axis
    i_y_left_mirror =  y[indices_left_mirror]
    i_y_right_mirror =  y[indices_right_mirror]

    i_x_left_mirror =  x[indices_left_mirror]
    i_x_right_mirror =  x[indices_right_mirror]





    # SUMMARY MIRROR DOMAIN ===========================================

    np.set_printoptions(threshold=np.inf)
    print("Indices left mirror: ", indices_left_mirror)
    print("Indices right mirror: ", indices_right_mirror)
    print("=======================================================\n")
    print("Initial_y_left_mirror: \n", i_y_left_mirror)
    print("Initial_y_right_mirror: \n", i_y_right_mirror)
    # print("Indices_all_check: ", indices_all)
    
    ave_initial_L = np.average(i_y_right_mirror) - np.average(i_y_left_mirror)
    print("Average Initial Optical Length: \n", ave_initial_L)
    print("=======================================================\n")


    #     x = dof_x[:, 0]
    #     print("DOFMAP: " , x)
    print("..............END TESTING EXTRACT NODE VALUES..............")


    
    

    # Boundary Conditions
    # x[2] refers to the z coordinate 
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[2] < DOLFIN_EPS
    dirichlet_clamped_boundary = DirichletBC(V, Constant((0.0, 0.0, 0.0)), clamped_boundary)

    # Define strain and stress
    def epsilon(u):
        engineering_strain = 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        return engineering_strain

    def sigma(u):
        cauchy_stress = (
            LAME_LAMBDA * tr(epsilon(u)) * Identity(3)
            +
            2 * LAME_MU * epsilon(u)
        )
        return cauchy_stress
    

    u_trial = TrialFunction(V)
    print("U_TRIAL: " , u_trial)
    v_test = TestFunction(V)
    forcing = Constant((0.0, 0.0, - DENSITY * ACCELERATION_DUE_TO_GRAVITY))
    traction = Constant((0.0, 0.0, 0.0))

    # another ex of LHS is a = dot(grad(u), grad(v))*dx 
    # another ex of RHS is L = f*v_test*dx 
    weak_form_lhs = inner(sigma(u_trial), epsilon(v_test)) * dx  # Crucial to use inner and not dot
    weak_form_rhs = (dot(forcing, v_test) * dx + dot(traction, v_test) * ds)

    # Compute solution
    # print(ufl_element(Function(V)))
    u_solution = Function(V)
#     print("FUNCTION(V): " , u_solution) 
#     print("Checking out u_solution vector before solving. : " + u_solution.vector())
    solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        dirichlet_clamped_boundary,
    )
    u_solution_vector = u_solution.vector().get_local()
    
    # =======================================================
    # ANALYSIS
    #
    # =======================================================

    u_displ_matrix = u_solution_vector.reshape(-1,3)
    print("=====================Solution=================")
    # print("u_displacement_matrix:\n", u_displ_matrix)

    # print("u_displ_y:\n", u_displ_matrix[:,1])
    u_displ_x = u_displ_matrix[:,0]
    u_displ_y = u_displ_matrix[:,1]
    print("len(u_displ_y):",len(u_displ_y))

    displ_x_left_mirror = u_displ_x[indices_left_mirror]
    displ_y_left_mirror = u_displ_y[indices_left_mirror]
    f_y_left_mirror = u_displ_y[indices_left_mirror] + i_y_left_mirror
    f_x_left_mirror = u_displ_x[indices_left_mirror] + i_x_left_mirror

    displ_y_right_mirror = u_displ_y[indices_right_mirror]
    displ_x_right_mirror = u_displ_x[indices_right_mirror]
    f_y_right_mirror = u_displ_y[indices_right_mirror] + i_y_right_mirror
    f_x_right_mirror = u_displ_x[indices_right_mirror] + i_x_right_mirror
    print("=======================================================\n")
    print("Final_y_left_mirror: \n", f_y_left_mirror)
    print("Final_y_right_mirror: \n", f_y_right_mirror)

    ave_final_L = np.average(f_y_right_mirror) - np.average(f_y_left_mirror)
    print("Average Final Optical Length: \n", ave_final_L)
    print("=======================================================\n")
    delta_L = np.abs(ave_final_L - ave_initial_L)
    
    plt.figure(figsize=(15,15))
    plt.xlim([0, N]); plt.ylim([0,N])
    plt.title("Mirror Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(i_x_left_mirror, i_y_left_mirror, 'k.')
    plt.plot(i_x_right_mirror, i_y_right_mirror, 'k.')

    plt.plot(f_x_left_mirror, f_y_left_mirror, 'r.')
    plt.plot(f_x_right_mirror, f_y_right_mirror, 'r.')

    plt.grid()
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    u_max = np.abs(u_solution_vector).max()
#     print("Maximum displacement: ", u_max)
    
    beam_deflection_file = XDMFFile(file_name + "_10.xdmf")
    beam_deflection_file.parameters["flush_output"] = True
    beam_deflection_file.parameters["functions_share_mesh"] = True
    beam_deflection_file.write(u_solution, 0.0)
    
    return [psi, u_max, delta_L]

# -------------------------------------------------------------------------------
# SIM FROM STATE
# -------------------------------------------------------------------------------

# given a state just run the solution and get the u_max
def sim_fromState(N, psi, mesh_res, file_name):

    # INITIALIZE START BOX
    # ===================
    startBox = Box(Point(0.0, 0.0, 0.0), Point(N,N,N))
    thisMesh = startBox

    # MODIFY MESH WITH STATE VECTOR 
    # ===================
    for index in range(N*N*N):
        if (psi[index] == 0):
            index_coord = getCoordFromIndex(index, N)
            # print("Removing index ", index, " with coords: ", index_coord[0], index_coord[1], index_coord[2])
            thisMesh = removeUnitCubeFromIndex(index, N, thisMesh)
        
    mesh = generate_mesh(thisMesh, 10)

    # lagrange_vector_space_first_order
    V = VectorFunctionSpace(
        mesh,
        "Lagrange",
        1,
    )
    
    
     # testing the extracting node values.
    print("..................TEST......................................")
    dofmap = V.dofmap()
    dofs = dofmap.dofs()

    # Get coordinates of all global indices in len(nodes) x 3 array. In fact we are getting 3 repeated coordinates for each one. 
    dofs_x = V.tabulate_dof_coordinates() 
    # print("Dofs_x:\n", dofs_x)
    # print("SHAPE OF DOFS_x:" , len(dofs_x))

    # x component of all the nodes. ORIGINAL COORIDNATES. 
    x = dofs_x[::3,0]; y = dofs_x[::3,1]; z = dofs_x[::3,2]
    print("len of x (initial from dofs_x):", len(x))


    # MIRROR DOMAIN ===========================================
    offset = 0.05
    indices_left_mirror = np.where((y > 1-offset) & (y < 1+offset) & (z > _ref) & (z < _ref+2) & (x > _ref) & (x < _ref+2))[0]
    indices_right_mirror = np.where((y > N-1 -0.05) & (y < N -1+ 0.05) & (z > _ref) & (z < _ref+2) & (x > _ref) & (x < _ref+2))[0]

    # we focus on the y direction cos thats the optical axis
    i_y_left_mirror =  y[indices_left_mirror]
    i_y_right_mirror =  y[indices_right_mirror]

    i_x_left_mirror =  x[indices_left_mirror]
    i_x_right_mirror =  x[indices_right_mirror]





    # SUMMARY MIRROR DOMAIN ===========================================

    np.set_printoptions(threshold=np.inf)
    print("Indices left mirror: ", indices_left_mirror)
    print("Indices right mirror: ", indices_right_mirror)
    print("=======================================================\n")
    print("Initial_y_left_mirror: \n", i_y_left_mirror)
    print("Initial_y_right_mirror: \n", i_y_right_mirror)
    # print("Indices_all_check: ", indices_all)
    
    ave_initial_L = np.average(i_y_right_mirror) - np.average(i_y_left_mirror)
    print("Average Initial Optical Length: \n", ave_initial_L)
    print("=======================================================\n")


    #     x = dof_x[:, 0]
    #     print("DOFMAP: " , x)
    print("..............END TESTING EXTRACT NODE VALUES..............")

    # Boundary Conditions
    # x[2] refers to the z coordinate 
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[2] < DOLFIN_EPS
    dirichlet_clamped_boundary = DirichletBC(V, Constant((0.0, 0.0, 0.0)), clamped_boundary)

    # Define strain and stress
    def epsilon(u):
        engineering_strain = 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        return engineering_strain

    def sigma(u):
        cauchy_stress = (
            LAME_LAMBDA * tr(epsilon(u)) * Identity(3)
            +
            2 * LAME_MU * epsilon(u)
        )
        return cauchy_stress
    

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)
    forcing = Constant((0.0, 0.0, - DENSITY * ACCELERATION_DUE_TO_GRAVITY))
    traction = Constant((0.0, 0.0, 0.0))

    # another ex of LHS is a = dot(grad(u), grad(v))*dx 
    # another ex of RHS is L = f*v_test*dx 
    weak_form_lhs = inner(sigma(u_trial), epsilon(v_test)) * dx  # Crucial to use inner and not dot
    weak_form_rhs = (dot(forcing, v_test) * dx + dot(traction, v_test) * ds)

    # Compute solution
    u_solution = Function(V)
#     print("Checking out u_solution vector before solving. : " + u_solution.vector())
    
    solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        dirichlet_clamped_boundary,
    )
    u_solution_vector = u_solution.vector().get_local()
    
    
    
    # =======================================================
    # ANALYSIS
    #
    # =======================================================

    u_displ_matrix = u_solution_vector.reshape(-1,3)
    print("=====================Solution=================")
    # print("u_displacement_matrix:\n", u_displ_matrix)

    # print("u_displ_y:\n", u_displ_matrix[:,1])
    u_displ_x = u_displ_matrix[:,0]
    u_displ_y = u_displ_matrix[:,1]
    print("len(u_displ_y):",len(u_displ_y))

    displ_x_left_mirror = u_displ_x[indices_left_mirror]
    displ_y_left_mirror = u_displ_y[indices_left_mirror]
    f_y_left_mirror = u_displ_y[indices_left_mirror] + i_y_left_mirror
    f_x_left_mirror = u_displ_x[indices_left_mirror] + i_x_left_mirror

    displ_y_right_mirror = u_displ_y[indices_right_mirror]
    displ_x_right_mirror = u_displ_x[indices_right_mirror]
    f_y_right_mirror = u_displ_y[indices_right_mirror] + i_y_right_mirror
    f_x_right_mirror = u_displ_x[indices_right_mirror] + i_x_right_mirror
    print("=======================================================\n")
    print("Final_y_left_mirror: \n", f_y_left_mirror)
    print("Final_y_right_mirror: \n", f_y_right_mirror)

    ave_final_L = np.average(f_y_right_mirror) - np.average(f_y_left_mirror)
    print("Average Final Optical Length: \n", ave_final_L)
    print("=======================================================\n")
    delta_L = np.abs(ave_final_L - ave_initial_L)
    
    plt.figure(figsize=(15,15))
    plt.xlim([0, N]); plt.ylim([0,N])
    plt.title("Mirror Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(i_x_left_mirror, i_y_left_mirror, 'k.')
    plt.plot(i_x_right_mirror, i_y_right_mirror, 'k.')

    plt.plot(f_x_left_mirror, f_y_left_mirror, 'r.')
    plt.plot(f_x_right_mirror, f_y_right_mirror, 'r.')

    plt.grid()
    plt.show()
    
    
    
    
    
    
    
    u_max = np.abs(u_solution_vector).max()
#     print("Maximum displacement: ", u_max)
    
    beam_deflection_file = XDMFFile(file_name + "_fromState" + "_10.xdmf")
    beam_deflection_file.parameters["flush_output"] = True
    beam_deflection_file.parameters["functions_share_mesh"] = True
    beam_deflection_file.write(u_solution, 0.0)
    
    return [u_max, delta_L]



def breed_crossover(p1, p2, crossover_pt, N):
    child = []
    for i in range(crossover_pt):
        child.append(p1[i])
        
    for j in range(N*N*N-crossover_pt):
        child.append(p2[crossover_pt + j])
    return (child)
        

def breed_random(p1, p2, N):
    child = []
    for i in range(N*N*N):
        choose = randrange(2)
#             print("k, Choose = ", k, ",", choose)
        if (choose == 0):
            child.append(p1[i])
        else:
            child.append(p2[i])
    return (child)


# ===========================================
# ===========================================
# MAIN CODE 
# ===========================================
# ===========================================

def main():
    # size of initial population
    nb_pop = 3; psi_array = []; u_max_array = []; delta_L_array =[];
    nb_breed = 5;
    N = 5
    offset = 3
    nb_chunks = 0
    maxSize_chunks = 2 
    mesh_res = 10
    
    # N = 15, mesh res = 20 just now. offset = 4. 
    # nb pop = 10, nb pop = 5; 
    start_time = time.time()
    
    # Initialize population 
    for i in range(nb_pop):
        print("current i: ", i)
        try: 
            print("--TRYING TO CREATE ", i)
            [psi, u_max, delta_L] = sim_randomChunks_hole(N, offset, nb_chunks, maxSize_chunks, "HORIZONTAL", mesh_res, "H_N5_260922_"+"parent_"+str(i))
            psi_array.append(psi); u_max_array.append(u_max); delta_L_array.append(delta_L)
            print(">>>>>> SUCCESSFUL >>>>>>")
        except: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!============== Parent failed. skip the i ==============!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            continue
#             [psi, u_max] = sim_randomChunks_hole(N, offset, 0, maxSize_chunks, "HORIZONTAL", mesh_res, str(i))
        
        # this should only come when it was successful
        print("PARENT ", i, " : \n", psi)
        print("Maximum displacement : ", u_max)
        print("delta_L : ", delta_L)

        print(" ----------------------------------------------------------------------- \n\n")
    
    print(".\n.\n.\n.\n.\n.\n.")
    # overwrite the size of the population, taking into account that some failed 
    nb_pop = len(psi_array)
    print(" ----------------------------------------------------------------------- \n\n")
    print("Initial population for genetic algorithm with size = ", nb_pop)
    print(" ----------------------------------------------------------------------- \n\n")

    for i in range(nb_pop):
#         print("Initial population for genetic algorithm")
        print("Parent ", i, " :", psi_array[i] )
        print("-------------------------------")
    
    
    
    print(".\n.\n.\n.\n.\n.\n.")
    for i in range(nb_breed):
        print("+++++++++++++++++++++++++++++",  "breeding" , "+++++++++++++++++++++++++++++")
        print("u_max_array for this iteration: \n" , u_max_array)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        child = []
        # choose random parent indices, ensure they are not the same parent
        p1_index = randrange(nb_pop)
        p2_index = randrange(nb_pop)
        count = 0
        while(p2_index == p1_index):
            p2_index = randrange(nb_pop)
            count += 1
            if (count > 10):
                break
        
        # get parent arrays from indices
        p1 = psi_array[p1_index]; p2 = psi_array[p2_index];
        
        
        # we need to find who is the worst parent for UMAX
        worst_uMax_parent_index = p1_index;
        
        if(u_max_array[p2_index] > u_max_array[p1_index]):
            worst_uMax_parent_index = p2_index;
        worst_u_max = u_max_array[worst_uMax_parent_index]
        
        # we need to find who is the worst parent for DELTA_L
        worst_L_parent_index = p1_index;
        
        if(delta_L_array[p2_index] > delta_L_array[p1_index]):
            worst_L_parent_index = p2_index;
        worst_L = delta_L_array[worst_L_parent_index]
#         print("Worst parent = worst_parent_index")

        # CREATE CHILD STATE VECTOR 
        # ==========================
        print("========== Parents of Child =========")
        print("P1: ", p1_index, "\nP2: ", p2_index)
        print("Worst parent for umax= ", worst_uMax_parent_index)
        print("Worst parent for delta L= ", worst_L_parent_index)
        print("=====================================")

        
        # crossover method. Take parents and crossover point. Creates a new child state vect. 
        child = breed_crossover(p1, p2, N*N, N)    
        print("Child: ", child)
        
        # if the child is not possible, I want to skip the child
        try: 
            print("--TRYING TO CREATE CHILD", i)
            [child_u_max, child_delta_L] = sim_fromState(N, child, mesh_res, "H_N20_130922_" + "breedSize_30_" +"child_" + str(i))
            print("--------Child is successfully simulated.--------")
            
            # if the child has a lesser deformation, we replace the worst parent in the population. 
            # otherwise, we forget about the child 
            # ================== UMAX CRITERIA ==================
#             if (child_u_max < worst_parent_u_max) :
#                 print("Child u_max VS Worst parent u_max: ", child_u_max, "," , worst_parent_u_max)
#                 print("Replacing worst parent with child")
#                 psi_array[worst_parent_index] = child
#                 u_max_array[worst_parent_index] = child_u_max
                
            # ================== DELTA L CRITERIA ==================

            if (child_delta_L < worst_parent_L) :
                print("Child delta_L VS Worst parent delta_L: ", child_delta_L, "," , worst_parent_L)
                print("Replacing worst parent with child")
                psi_array[worst_parent_index] = child
                delta_L_array[worst_parent_index] = child_delta_L
    
        except: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!============== Child failed. skip the i ==============!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            continue
        
    

    print("Time taken: %s seconds" % (time.time() - start_time))
    print("Done")

#     beam_deflection_file.write(von_Mises_stress, 0.0)












if __name__ == "__main__":
    main()
