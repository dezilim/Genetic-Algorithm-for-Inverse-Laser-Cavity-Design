from fenics import *
from dolfin import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import math 
from random import randrange

from mshr import *


import time
np.set_printoptions(threshold=np.inf)


# try:
#     import pyvista
# except ModuleNotFoundError:
#     print("pyvista is required for this demo")
#     exit(0)

# try:
#     pyvista.set_jupyter_backend('ipyvtklink')
# except ModuleNotFoundError:
#     print("pythreejs is required for this demo")
#     exit(0)

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
ACCELERATION_DUE_TO_GRAVITY = 0.008








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
# - psi_removeVerticalHole(N, psi_arr): turns off all indices along a 2 by 2 vertical block
# -
# ============================================================================

def psi_removeVerticalHole(N, psi_arr):
    # if even, remove a simple 4 block 

    _ref = np.floor((N-1)/2)
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
#     if (N%2 == 0):
#         # bottom left coord
#         _ref = N/2 -1
        
#     else: 
#         _ref = (N-1)/2
    _ref = np.floor((N-1)/2)
        
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

                    
    return (psi_arr)
                
# create and return psi, representing just the bare mimimum to create a hole with material around it 
# you can input a psi_arr that already has chunks; it will not mess with those except make sure there is at least material around a hole
def createHorizontalHole(N, psi_arr):
#     psi_arr = np.array([0] * N*N*N)
    # simplify it, if its odd just shift it slightly to the left 
#     if (N%2 == 0):
#         # bottom left coord
#         _ref = N/2 -1
        
#     else: 
#         _ref = (N-1)/2
        
    _ref = np.floor((N-1)/2)
        
    print("Ref for psi_removeVerticalHole: ", _ref)
    # index without the z part. (x+1) + y*N 
    print("REF: BL, BR, TL, TR:") 
    BL = _ref + _ref*N*N 
    BR = (_ref + 1) + _ref*N*N
    TL = _ref + (_ref+1)*N*N 
    TR = (_ref + 1) + (_ref+1)*N*N 
    print(BL,  "\n",BR, "\n", TL, "\n", TR, "\n")


    for y in np.arange(0,N):
        if ((y == 0) or (y ==  N-1)):
            psi_arr[int(BL + y*N)] = 1
            psi_arr[int(BR + y*N)] = 1
            psi_arr[int(TL + y*N)] = 1
            psi_arr[int(TR + y*N)] = 1
            
        else:   
            psi_arr[int(BL + y*N)] = 0
            psi_arr[int(BR + y*N)] = 0
            psi_arr[int(TL + y*N)] = 0
            psi_arr[int(TR + y*N)] = 0

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
    if (index >= N*N*N):
        return (psi_arr)
    
    psi_arr[int(index)] = 0
    return (psi_arr)

def psi_addCube(N, psi_arr, xi, yi, zi):
    index = getIndexFromCoord(xi, yi, zi, N)
    if (index >= N*N*N):
        return (psi_arr)
    
    psi_arr[int(index)] = 1
    return (psi_arr)

# square chunk
# obviously, you cannot have xi yi or zi less than 0
def psi_removeChunk(N, psi_arr, xi, yi, zi, size):
#     if ((xi + size > N) or (yi + size > N) or (zi + size > N)):
#         raise Exception("Chunk to remove excceeds bounds")
        
    for _z in np.arange(zi, zi + size):
        for _y in np.arange(yi, yi + size):
            for _x in np.arange(xi, xi + size):
                psi_arr = psi_removeCube(N, psi_arr, _x,_y,_z)
    return (psi_arr)

# square chunk
def psi_addChunk(N, psi_arr, xi, yi, zi, size):
#     if ((xi + size > N) or (yi + size > N) or (zi + size > N)):
#         raise Exception("Chunk to remove excceeds bounds")
        
    for _z in np.arange(zi, zi + size):
        for _y in np.arange(yi, yi + size):
            for _x in np.arange(xi, xi + size):
                psi_arr = psi_addCube(N, psi_arr, _x,_y,_z)
    return (psi_arr)

# chunk of size size_x, size_y, size_z
def psi_addChunkFree(N, psi_arr, xi, yi, zi, size_x, size_y, size_z):
            
    for _z in np.arange(zi, zi + size_z):
        for _y in np.arange(yi, yi + size_y):
            for _x in np.arange(xi, xi + size_x):
                psi_arr = psi_addCube(N, psi_arr, _x,_y,_z)
    return (psi_arr)



# -------------------------------------------------------------------------------
# SIM FROM RANDOM CHUNKS HOLE
# -------------------------------------------------------------------------------

# Simulate a configuration with randomly chosen chunks with maximum sizes. Hole state can be horizontal or vertical. 
# Mesh res is for creating the mesh. Use 10 for N = 10
# maxSize_chunks should be less than the offset
def sim_randomChunks_hole(N, offset, nb_chunks, maxSize_chunks, design, hole_state, mesh_res, file_name):
    nb_voxels = N*N*N
    nb_empty = 0
    
    _ref = np.floor((N-1)/2)
        

# ================================================
# subtractive design. Removes chunks then hole with mirrors.
# remove smaller chunks -> more chance of connected structure 
# ================================================
    if (design == "subtractive"):
        print("Using subtractive method")
        
        # start with a full state vector
#         psi = np.array([0] * nb_empty + [1] * (nb_voxels - nb_empty))
        psi = np.array([1] * N*N*N)

        # remove some chunks from the statevect
        # =====================================
        padding = 0 # padding must be less than offset
        for i in range(nb_chunks):
            # get a random x,y,z coord and size to remove 
            
            size = randrange(maxSize_chunks);
            xi = randrange(N-size) + padding ; yi = randrange(N-size) + padding; zi = randrange(N-size) + 1 + padding; 
            print("Chosen (x,y,z, size): " , xi, yi, zi, size)
            psi = psi_removeChunk(N, psi, xi, yi, zi, size)
            print("=============================")


        # If none of these then dont remove hole
        if (hole_state == "HORIZONTAL"):
            psi = psi_removeHorizontalHole(N, psi)
        elif (hole_state == "VERTICAL"):
            psi = psi_removeVerticalHole(N, psi)
      
    
    # CONSTRAINT FOR THE SUPPORT 
#     # ===================
#     # REMOVE ENTIRE FIRST LAYER FIRST 
#     psi[0:N*N] = [0]*N*N



# ================================================
# generative design
# add bigger chunks -> more chance of connected structure
# ================================================
    elif (design == "generative"):
        
        
        
        
        psi = np.array([0] * N*N*N)
            # ADD some chunks from the statevect
        # =====================================
        padding = 0 # padding must be less than offset
        for i in range(nb_chunks):
            # get a random x,y,z coord and size to remove 
            # add a +1 for the z as I am testing with the support and I don't want chunks to appear on the bottommost layer
            
            size = randrange(maxSize_chunks) +3;
            xi = randrange(N-size) + padding ; yi = randrange(N-size) + padding; zi = randrange(N-size) + 1 + padding; 
            print("Chosen (x,y,z, size): " , xi, yi, zi, size)
            psi = psi_addChunk(N, psi, xi, yi, zi, size)
            print("=============================")


        psi = createHorizontalHole(N, psi)

        # create a pillars
        nb_supports = 3
        supp_thickness = 2
        supp_height = int(np.floor(N/2))
        center_ref = int(np.floor(N/2))
        psi = psi_addChunkFree(N, psi, center_ref-4, 1, 0, supp_thickness, supp_thickness, supp_height)
        psi = psi_addChunkFree(N, psi, center_ref+3, 1, 0, supp_thickness, supp_thickness, supp_height)
        psi = psi_addChunkFree(N, psi, center_ref-1, N-1-supp_thickness, 0, supp_thickness, supp_thickness, supp_height)

    
#     center_ref = int(np.floor(N/2))
#     for z in range(center_ref - 1):
#         psi[(center_ref-1)+ 1*N + z*N*N] = 1
#         psi[(center_ref-1)+ 2*N + z*N*N] = 1
#         psi[(center_ref-1) + (N-2)*N + z*N*N] = 1
#         psi[(center_ref-1) + (N-3)*N + z*N*N] = 1
#         psi[center_ref+ 1*N + z*N*N] = 1
#         psi[center_ref+ 2*N + z*N*N] = 1
#         psi[center_ref + (N-2)*N + z*N*N] = 1
#         psi[center_ref + (N-3)*N + z*N*N] = 1

        
#     print("resultant psi: ", psi)
    # ===================

    
    
    
    
    
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
    indices_left_mirror = np.where((y > 1-offset) & (y < 1+offset) & (z > _ref - offset) & (z < _ref+2 + offset) & (x > _ref - offset) & (x < _ref+2 + offset))[0]
    indices_right_mirror = np.where((y > N-1 -offset) & (y < N -1+ offset) & (z > _ref - offset) & (z < _ref+2 + offset) & (x > _ref - offset) & (x < _ref+2 + offset))[0]

    # we focus on the y direction cos thats the optical axis
    i_y_left_mirror =  y[indices_left_mirror]
    i_y_right_mirror =  y[indices_right_mirror]

    i_x_left_mirror =  x[indices_left_mirror]
    i_x_right_mirror =  x[indices_right_mirror]





    # SUMMARY MIRROR DOMAIN ===========================================

    np.set_printoptions(threshold=np.inf)
#     print("Indices left mirror: ", indices_left_mirror)
#     print("Indices right mirror: ", indices_right_mirror)
    print("=======================================================\n")
#     print("Initial_y_left_mirror: \n", i_y_left_mirror)
#     print("Initial_y_right_mirror: \n", i_y_right_mirror)
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
    u_max = np.abs(u_solution_vector).max()
    print("u_max: ")
    print(u_max)
    if ((math.isnan(u_max)) or (int(u_max) > 8)):
        raise Exception("Nonsensical Result, failed analysis")
    # =======================================================
    # ANALYSIS
    #
    # =======================================================

    u_displ_matrix = u_solution_vector.reshape(-1,3)
#     print("=====================Solution=================")
#     print("u_displacement_matrix:\n", u_displ_matrix)

    # print("u_displ_y:\n", u_displ_matrix[:,1])
    u_displ_x = u_displ_matrix[:,0]
    u_displ_y = u_displ_matrix[:,1]
    u_displ_z = u_displ_matrix[:,2]
#     print("len(u_displ_y):",len(u_displ_y))

    displ_x_left_mirror = u_displ_x[indices_left_mirror]
    displ_y_left_mirror = u_displ_y[indices_left_mirror]
    f_y_left_mirror = u_displ_y[indices_left_mirror] + i_y_left_mirror
    f_x_left_mirror = u_displ_x[indices_left_mirror] + i_x_left_mirror

    displ_y_right_mirror = u_displ_y[indices_right_mirror]
    displ_x_right_mirror = u_displ_x[indices_right_mirror]
    f_y_right_mirror = u_displ_y[indices_right_mirror] + i_y_right_mirror
    f_x_right_mirror = u_displ_x[indices_right_mirror] + i_x_right_mirror
    print("=======================================================\n")
#     print("Final_y_left_mirror: \n", f_y_left_mirror)
#     print("Final_y_right_mirror: \n", f_y_right_mirror)

    ave_final_L = np.average(f_y_right_mirror) - np.average(f_y_left_mirror)
    print("Average Final Optical Length: \n", ave_final_L)
    print("=======================================================\n")
    delta_L = np.abs(ave_final_L - ave_initial_L)
    
    
    plt.figure(figsize=(5,5))
    plt.xlim([0, N]); plt.ylim([0,N])
    plt.title("Mirror Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(i_x_left_mirror, i_y_left_mirror, 'k.', label = "initial")
    plt.plot(i_x_right_mirror, i_y_right_mirror, 'k.')

    plt.plot(f_x_left_mirror, f_y_left_mirror, 'r.', label = "final")
    plt.plot(f_x_right_mirror, f_y_right_mirror, 'r.')

    plt.grid()
    
    

# plot the quiver displacements 
#     fig = plt.figure(figsize = (15,10))
#     ax = fig.gca(projection='3d')
#     scaled_dx = [2*i for i in u_displ_x]
#     scaled_dy = [2*i for i in u_displ_y]
#     scaled_dz = [2*i for i in u_displ_z]
#     ax.quiver(x,y,z, scaled_dx, scaled_dy, scaled_dz, color='red')
    
#     # plot of horizontal plane
#     X, Y = np.meshgrid(np.arange(0, N+1), np.arange(0, N+1))
#     Z = 0*X
#     ax.plot_surface(X, Y, Z, alpha=0.3)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
    plt.legend()
    plt.show()
    
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(x, y, z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
#     ax.set_title('surface');

#     ax.scatter(x,y,z)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
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

#     if (N%2 == 0):
#         # bottom left coord
#         _ref = N/2 -1
        
#     else: 
#         _ref = (N-1)/2
    _ref = np.floor((N-1)/2)
    
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
    indices_left_mirror = np.where((y > 1-offset) & (y < 1+offset) & (z > _ref - offset) & (z < _ref+2 + offset) & (x > _ref - offset) & (x < _ref+2 + offset))[0]
    indices_right_mirror = np.where((y > N-1 -offset) & (y < N -1+ offset) & (z > _ref - offset) & (z < _ref+2 + offset) & (x > _ref - offset) & (x < _ref+2 + offset))[0]

    # we focus on the y direction cos thats the optical axis
    i_y_left_mirror =  y[indices_left_mirror]
    i_y_right_mirror =  y[indices_right_mirror]

    i_x_left_mirror =  x[indices_left_mirror]
    i_x_right_mirror =  x[indices_right_mirror]





    # SUMMARY MIRROR DOMAIN ===========================================

    np.set_printoptions(threshold=np.inf)
#     print("Indices left mirror: ", indices_left_mirror)
#     print("Indices right mirror: ", indices_right_mirror)
    print("=======================================================\n")
#     print("Initial_y_left_mirror: \n", i_y_left_mirror)
#     print("Initial_y_right_mirror: \n", i_y_right_mirror)
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
    
    u_max = np.abs(u_solution_vector).max()
    print("u_max: ")
    print(u_max)
    if ((math.isnan(u_max)) or (int(u_max) > 8)):
        raise Exception("Nonsensical Result, failed analysis")
    
    # =======================================================
    # ANALYSIS
    #
    # =======================================================

    u_displ_matrix = u_solution_vector.reshape(-1,3)
    print("=====================Solution=================")
#     print("u_displacement_matrix:\n", u_displ_matrix)

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
#     print("Final_y_left_mirror: \n", f_y_left_mirror)
#     print("Final_y_right_mirror: \n", f_y_right_mirror)

    ave_final_L = np.average(f_y_right_mirror) - np.average(f_y_left_mirror)
    print("Average Final Optical Length: \n", ave_final_L)
    print("=======================================================\n")
    delta_L = np.abs(ave_final_L - ave_initial_L)
    
    plt.figure(figsize=(5,5))
    plt.xlim([0, N]); plt.ylim([0,N])
    plt.title("Mirror Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(i_x_left_mirror, i_y_left_mirror, 'k.', label = "initial")
    plt.plot(i_x_right_mirror, i_y_right_mirror, 'k.')

    plt.plot(f_x_left_mirror, f_y_left_mirror, 'r.', label = "final")
    plt.plot(f_x_right_mirror, f_y_right_mirror, 'r.')

    plt.grid()
    
    

    # plot the quiver displacements

#     fig = plt.figure(figsize = (15,10))
#     ax = fig.gca(projection='3d')
#     scaled_dx = [2*i for i in u_displ_x]
#     scaled_dy = [2*i for i in u_displ_y]
#     scaled_dz = [2*i for i in u_displ_z]
#     ax.quiver(x,y,z, scaled_dx, scaled_dy, scaled_dz, color='red')
    
#     # plot of horizontal plane
#     X, Y = np.meshgrid(np.arange(0, N+1), np.arange(0, N+1))
#     Z = 0*X
#     ax.plot_surface(X, Y, Z, alpha=0.3)
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
#     print("Maximum displacement: ", u_max)
    
    beam_deflection_file = XDMFFile(file_name + "_fromState" + "_10.xdmf")
    beam_deflection_file.parameters["flush_output"] = True
    beam_deflection_file.parameters["functions_share_mesh"] = True
    beam_deflection_file.write(u_solution, 0.0)
    
    return [u_max, delta_L]



def breed_crossover1(p1, p2, crossover_pt1, N, invert):
    child = []
#     child_front = p1[0:crossover_pt1]
#     print("FRONT:", child_front)
#     child_middle = p2[crossover_pt1:crossover_pt2]
#     print("MIDDLE:", child_middle)

#     child_end = p1[crossover_pt2:N]
    
#     child = child_front + child_middle + child_end
    for i in range(crossover_pt1):
        child.append(p1[i])
        
    for j in range(N*N*N-crossover_pt1):
        child.append(p2[crossover_pt1 + j])
        
    if (invert == 1):
        print("-- Crossover breed inverted --")
        child = breed_invert(child)
        
    return (child)



def breed_crossover2(p1, p2, crossover_pt1, crossover_pt2, N, invert):
    child = []
#     child_front = p1[0:crossover_pt1]
#     print("FRONT:", child_front)
#     child_middle = p2[crossover_pt1:crossover_pt2]
#     print("MIDDLE:", child_middle)

#     child_end = p1[crossover_pt2:N]
    
#     child = child_front + child_middle + child_end
    for i in range(crossover_pt1):
        child.append(p1[i])
        
    for j in range(crossover_pt2-crossover_pt1):
        child.append(p2[crossover_pt1 + j])
        
    for k in range(N*N*N-crossover_pt2):
        child.append(p2[crossover_pt2 + k])
    if (invert == 1):
        print("-- Crossover breed inverted --")
        child = breed_invert(child)
        
    return (child)
        
def breed_invert(state):
    state = [1-x for x in state]
    return (state)

def breed_mutate(state, N, nb_mutations, mutation_indices):
    for i in range(nb_mutations):
        # randomly choose an index in the list of indices possible to mutate
        rand_index = randrange(len(mutation_indices)) 
        mutation_index = mutation_indices[rand_index]
        if (mutation_index == 0):
            return(state)
        state[mutation_index] = 1- state[mutation_index]
    return (state)
    
    
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
# random corssover. np pop 20 , nb breed 100 , nbchunks 15
def main():
    # size of initial population
    nb_pop = 20; psi_array = []; u_max_array = []; delta_L_array =[];
    nb_breed = 100; 
    N = 12
    offset = 6 # not really used anymore.
    nb_chunks = 30 # i use 5 for subtractive and 8 for generative 
    maxSize_chunks = 3 # not so constrained anymore. Ensure less than N
    mesh_res = 10
    
    
    
    print("%%%%%%%%%%%%%%% DETAILS OF RUN %%%%%%%%%%%%%%%")
    print("nb_pop: %s\nnb_breed: %s\nN: %s\noffset: %s\nnb_chunks: %s\nmaxSize_chunks: %s\nmesh_res: %s\n" % (nb_pop, nb_breed, N, offset, nb_chunks, maxSize_chunks, mesh_res))
    print("%%%%%%%%%%%%%%% DETAILS OF RUN %%%%%%%%%%%%%%%")

    
    # N = 15, mesh res = 20 just now. offset = 4. 
    # nb pop = 10, nb pop = 5; 
    start_time = time.time()
    
    # Initialize population 
    for i in range(nb_pop):
        print("current i: ", i)
        try: 
            print("--TRYING TO CREATE ", i)
            [psi, u_max, delta_L] = sim_randomChunks_hole(N, offset, nb_chunks, maxSize_chunks, "subtractive", "HORIZONTAL", mesh_res, "SUB_CO2_H_supp_N12_c30_"+"parent_"+str(i))
            psi_array.append(psi); u_max_array.append(u_max); delta_L_array.append(delta_L)
            print(">>>>>> SUCCESSFUL >>>>>>")
        except: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!============== Parent failed. skip the i ==============!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            continue
#             [psi, u_max] = sim_randomChunks_hole(N, offset, 0, maxSize_chunks, "HORIZONTAL", mesh_res, str(i))
        
        # this should only come when it was successful
#         print("PARENT ", i, " : \n", psi)
        print("PARENT ", i)
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
    winner_index = [-1]*nb_pop
    
    child_count = 0
    for i in range(nb_breed):
        
        if (i>10):
            if(max(u_max_array) < 0.001):
                break
        
        
        print("+++++++++++++++++++++++++++++",  "breeding" , "+++++++++++++++++++++++++++++")
        print("u_max_array for this iteration: \n" , u_max_array)
        print("delta_L_array for this iteration: \n" , delta_L_array)
        print("winner_index_array for this iteration: \n," , winner_index)

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
#         print("========== Parents of Child =========")
#         print("P1: ", p1_index, "\nP2: ", p2_index)
#         print("Worst parent for umax= ", worst_uMax_parent_index)
#         print("Worst parent for delta L= ", worst_L_parent_index)
#         print("=====================================")

        
        # crossover method. Take parents and crossover point. Creates a new child state vect. 
        
#         # N = 10
#         crossoverpt1 = 200;
#         crossoverpt2 = 800;   
        
        # N = 12
        crossoverpt1 = 300;
        crossoverpt2 = 800;  
#         crossoverpt = randrange(N*N*N-N*N) + N*N
        
        
#         # N = 20
#         crossoverpt1 = 2500;
#         crossoverpt2 = 6000;          
        
#         # N = 20
#         crossoverpt1 = 1000;
#         crossoverpt2 = 2000;  
        
#         # N = 5
#         crossoverpt1 = 25;
#         crossoverpt2 = 75;
        invert = randrange(1); # this will help decide which of the two possible children we should have. 
        nb_mutations = randrange(10);
        print("========== Child creation info =========")
        print("P1: ", p1_index, "\nP2: ", p2_index)
        print("Worst parent for umax : ", worst_uMax_parent_index)
        print("Worst parent for delta L : ", worst_L_parent_index)
        print("(Crossoverpt1, Crossoverpt2) : " , crossoverpt1, ", " , crossoverpt2)
#         print("Random Crossoverpoint Chosen:", crossoverpt)
        print("Invert ? : ", invert)
        print("Nb_mutations : ", nb_mutations)
        print("========================================")
        
        # one point crossover
#         child = breed_crossover1(p1, p2, crossoverpt, N, invert)    
        
        # two point crossover 
        child = breed_crossover2(p1, p2, crossoverpt1, crossoverpt2, N, invert)    
#         print("Child: ", child)
        # mutate
        # mutation_indices gives the elements that you can possibily mutate. We dont want to mutate the bottomost and top most layer
        # as well as the side layers parallel to the hole
        mutation_indices = list(range(N*N*N))
        mutation_indices[:N*N]  = [0]*N*N
        mutation_indices[N*N*N-N*N:N*N*N] = [0]*(N*N)
        for j in range(len(mutation_indices)):
            if (mutation_indices[j]%N == 0) or (mutation_indices[j]%N == N-1):
                mutation_indices[j] = 0
                
        child = breed_mutate(child, N, nb_mutations, mutation_indices)
#         print("Child after mutation:", child)
        
        # if the child is not possible, I want to skip the child
        try: 
            print("--TRYING TO CREATE CHILD", i)
            [child_u_max, child_delta_L] = sim_fromState(N, child, mesh_res, "SUB_CO2_H_supp_N12_c30_" + "breedSize_100_" +"child_" + str(i))
            print("--------Child is successfully simulated.--------")
            child_count += 1
            
            # if the child has a lesser deformation, we replace the worst parent in the population. 
            # otherwise, we forget about the child 
            # ================== UMAX CRITERIA ==================
#             if (child_u_max < worst_parent_u_max) :
#                 print("Child u_max VS Worst parent u_max: ", child_u_max, "," , worst_parent_u_max)
#                 print("Replacing worst parent with child")
#                 psi_array[worst_parent_index] = child
#                 u_max_array[worst_parent_index] = child_u_max
                
            # ================== DELTA L CRITERIA ==================

            if (child_delta_L < worst_L) :
                print("Child delta_L VS Worst parent delta_L: ", child_delta_L, "," , worst_L)
                print("Replacing worst parent with child")
                psi_array[worst_L_parent_index] = child
                u_max_array[worst_L_parent_index] = child_u_max
                delta_L_array[worst_L_parent_index] = child_delta_L
                winner_index[worst_L_parent_index] = i 
            print("-------------------END BREED SESSION-------------------")
            print("-------------------------------------------------------")
    
        except: 
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!============== Child failed. skip the i ==============!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            continue
            
    
    print("-------------------SUMMARY OF RUN----------------------")
    print("-------------------------------------------------------")
    print("u_max_array : \n" , u_max_array)
    print("delta_L_array : \n" , delta_L_array)
    print("winner_index_array : \n," , winner_index)
    print("Child count: \n,", child_count)
    print("Time taken: %s seconds" % (time.time() - start_time))
    print("Done")

#     beam_deflection_file.write(von_Mises_stress, 0.0)












if __name__ == "__main__":
    main()
