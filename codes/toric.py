import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, eye_array, block_array

from copy import deepcopy


"""
Toric code of distance d:
    Suppose d = 4

             |         |         |         |
             o         o         o         o
             |         |         |         |
    --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |
             o         o         o         o
             |         |         |         |
    --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |
             o         o         o         o
             |         |         |         |
    --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |
             o         o         o         o
             |         |         |         |
    --- o --- --- o --- --- o --- --- o --- 

    X stabilizers: vertices
    Z stabilizers: plaquettes

    For distance d, there are 2 * d^2 qubits. We order the qubits based on the vertex
    to the right or below. 
"""

"""
We construct a csr (compressed sparse row) with 
`csr_array((data, indices, indptr), shape=(M, N), dtype=np.int8).toarray()`
    indices - list of all column indices for each row, concatenated
    indptr[i] - where the column indices for row i starts; this is one to the right of
        where the indices for row i-1 end
    data - values for indices
"""

def toric_code_symplectic(L):
    I = eye_array(m=2*L*L, dtype=np.int8, format='csr')
    return block_array([[None, I], [I, None]], format='csr', dtype=np.int8)

def toric_code_x(L, verbose=False):
    """
    Parity check for toric code with k=2 logical qubits; only X stabilizers
    """
    assert L > 1
    indices = []
    indptr = [0]

    # L^2 vertex stabilizers

    def i_to_xy(i):
        # i is the vertex stabilizer index
        # Get row and column indices for vertices
        return i // L, i % L
    def xy_to_i(x, y):
        # Convert row and column indices to stabilizer index
        return x*L + y

    if verbose:
        print('Generate X stabilizers')
    for i in range(L**2):
        # i is the vertex index
        x, y = i_to_xy(i)

        i_below = xy_to_i(x, (y-1) % L)
        i_right = xy_to_i((x+1) % L, y)

        # left, up, down, right
        l_ind = [2*i, 2*i+1, 2*i_below+1, 2*i_right]
        if verbose:
            print(l_ind)

        indices.extend(l_ind)
        indptr.append(indptr[-1]+4)
    
    data = np.ones(len(indices), dtype=np.int8)
    return csr_matrix((data, indices, indptr), shape=(L**2, 2*L**2), dtype=np.int8)

def toric_code_z(L, verbose=False):
    """
    Parity check for toric code with k=2 logical qubits; only Z stabilizers
    """
    assert L > 1
    indices = []
    indptr = [0]

    # L^2 vertex stabilizers

    def i_to_xy(i):
        # i is the vertex stabilizer index
        # Get row and column indices for vertices
        return i // L, i % L
    def xy_to_i(x, y):
        # Convert row and column indices to stabilizer index
        return x*L + y

    if verbose:
        print('Generate Z stabilizers')
    for i in range(L**2):
        # i is the vertex index
        x, y = i_to_xy(i)

        i_above = xy_to_i(x, (y+1) % L)
        i_left = xy_to_i((x-1) % L, y)

        # left, up, down, right
        l_ind = [2*i, 2*i+1, 2*i_above, 2*i_left+1]
        if verbose:
            print(l_ind)

        indices.extend(l_ind)
        indptr.append(indptr[-1]+4)
    
    data = np.ones(len(indices), dtype=np.int8)
    return csr_matrix((data, indices, indptr), shape=(L**2, 2*L**2), dtype=np.int8)
    
def toric_code(L, verbose=False):
    """
    Parity check for toric code with k=2 logical qubits; both X and Z stabilziers
    """
    H_X = toric_code_x(L, verbose=verbose)
    H_Z = toric_code_z(L, verbose=verbose)
    H = block_array([[H_X, None], [None, H_Z]], format='csr', dtype=np.int8)
    assert np.allclose((H @ toric_code_symplectic(L) @ H.T).data % 2, 0)
    return H

def toric_code_logical_z(L, verbose=False):
    """
    Logical z operators; we place these at the bottom and left of the grid
    """
    assert L > 1
    indices = []
    indptr = [0]

    if verbose:
        print('Generate Z logicals')

    # Horizontal Z operator through prime lattice
    l_ind = [2*L*i for i in range(L)]
    if verbose:
            print(l_ind)
    indices.extend(l_ind)
    indptr.append(indptr[-1]+L)
    
    # Vertical Z operator through prime lattice
    l_ind = [2*i + 1 for i in range(L)]
    if verbose:
            print(l_ind)
    indices.extend(l_ind)
    indptr.append(indptr[-1]+L)

    data = np.ones(len(indices), dtype=np.int8)
    
    return csr_matrix((data, indices, indptr), shape=(2, 2*L**2), dtype=np.int8)

def toric_code_logical_x(L, verbose=False):
    """
    Logical x operators; we place these at the bottom and left of the grid
    """
    assert L > 1
    indices = []
    indptr = [0]

    if verbose:
        print('Generate X logicals')

    # Horizontal x operator through dual lattice
    l_ind = [2*L*i +1 for i in range(L)]
    if verbose:
            print(l_ind)
    indices.extend(l_ind)
    indptr.append(indptr[-1]+L)
    
    # Vertical X operator through dual lattice
    l_ind = [2*i for i in range(L)]
    if verbose:
            print(l_ind)
    indices.extend(l_ind)
    indptr.append(indptr[-1]+L)

    data = np.ones(len(indices), dtype=np.int8)
    return csr_matrix((data, indices, indptr), shape=(2, 2*L**2), dtype=np.int8)

def toric_code_logical(L, verbose=False):
    """
    All four logical operators for toric code on torus
    """
    L_X = toric_code_logical_x(L, verbose=verbose)
    L_Z = toric_code_logical_z(L, verbose=verbose)
    Log = block_array([[L_X, None], [None, L_Z]], format='csr', dtype=np.int8)

    k = 2
    assert np.all((Log @ toric_code_symplectic(2) @ Log.T).indptr == np.arange(2*k + 1))
    assert np.all((Log @ toric_code_symplectic(2) @ Log.T).indices == np.arange(2*k)[::-1])
    
    return Log

"""
Surface code of distance d with k logical qubits:
    Suppose d = 4 and k = 1

    --- o --- --- o --- --- o --- --- o ---
             |         |         |         
             o         o         o         
             |         |         |         
    --- o --- --- o --- --- o --- --- o ---
             |         |         |         
             o         o         o         
             |         |         |         
    --- o --- --- o --- --- o --- --- o ---
             |         |         |         
             o         o         o         
             |         |         |         
    --- o --- --- o --- --- o --- --- o ---
 
                                 
    Suppose d = 4 and k = 2

                                 |         |         |         |
                                 o         o         o         o
                                 |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |
             o         o         o         o         o         o
             |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |
             o         o         o         o         o         o
             |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |
             o         o         o         o         o         o
             |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
                                 |         |         |         |
                                 o         o         o         o
                                 |         |         |         |
                                 
    Suppose d = 4 and k = 3

                                 |         |         |         |
                                 o         o         o         o
                                 |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o ---
             |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o ---
             |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o ---
                                 |         |         |         |
                                 o         o         o         o
                                 |         |         |         |

    Suppose d = 4 and k = 4

                                 |         |         |         |                   |         |         |         |
                                 o         o         o         o                   o         o         o         o
                                 |         |         |         |                   |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
                                 |         |         |         |                   |         |         |         |
                                 o         o         o         o                   o         o         o         o
                                 |         |         |         |                   |         |         |         |

    Suppose d = 4 and k = 5

                                 |         |         |         |                   |         |         |         |
                                 o         o         o         o                   o         o         o         o
                                 |         |         |         |                   |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
             |         |         |         |         |         |         |         |         |         |         |         |         |
             o         o         o         o         o         o         o         o         o         o         o         o         o
             |         |         |         |         |         |         |         |         |         |         |         |         |
    --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- --- o --- 
                                 |         |         |         |                   |         |         |         |
                                 o         o         o         o                   o         o         o         o
                                 |         |         |         |                   |         |         |         |

Note that if k > 1, we will have rough boundary conditions somewhere at the top and bottom of the grid. Thus
"""

def surface_code_x(L, k, verbose=False):
    """
    Parity check for (unrotated) surface code with k logical qubits; only X stabilizers

    """
    assert L > 3
    
    if k == 1:
        Lx = L-1
    elif k % 2 == 1:
        Lx = 2*(L-2) # Left and right rough boundaries
        Lx += (k-2)//2 * (L+L-3) # For each rough - smooth alternation, there are L+1 columns
        Lx += L # Last rough
    elif k % 2 == 0:
        Lx = (L-2) # Left rough boundaries
        Lx += (k-1)//2 * (L+L-3) # For each rough - smooth alternation, there are L+1 columns
        Lx += L # Last rough
    else:
        assert False
    Ly = L

    # boundary_y is a size Lx Boolean array. For each column, is the boundary condition above (or below; they are the same).
    # boundary_x is a size 2 Boolean array. The BCs of each row on the left are the same; the right BCs of each row are the 
    # same as one another but can be different from the left.
    # True is smooth, False is rough
    boundary_y = np.zeros(Lx, dtype=bool)
    if k == 1:
        boundary_y[:] = True
    elif k % 2 == 1:
        boundary_y[:L-2] = True
        boundary_y[-(L-2):] = True
        boundary_y[L-2:-(L-2)] = ([False]*L + [True]*(L-3)) * ((k-2)//2) + [False]*L
    elif k % 2 == 0:
        boundary_y[:L-2] = True
        boundary_y[L-2:] = ([False]*L + [True]*(L-3)) * ((k-2)//2) + [False]*L

    boundary_x = np.zeros(2, dtype=bool)
    boundary_x[:] = [False, (k % 2) == 0]
    
    print(Lx, Ly)
    print(boundary_x, boundary_y)

    num_qubits = 0
    seen_qubits = []
    for i, by in enumerate(boundary_y):
        # Assign qubit left and above stabilizer to the vertex
        # Rough boundary: add bottom qubit
        # Smooth boundary: remove top qubit
        seen_qubits.append(2 * Ly + (-1 if by else 1))
        #num_qubits += 2 * Ly + (-1 if by else 1)

    # Ly qubits on right boundary if rough
    if not boundary_x[1]:
        seen_qubits[-1] += Ly
        #num_qubits += Ly
    num_qubits = np.sum(seen_qubits)
    print(num_qubits)

    indices = []
    indptr = [0]

    if verbose:
        print('Generate X stabilizers')
        
    for i in range(Lx):
        qind_i = int(np.sum(seen_qubits[:i]))
        print(qind_i, seen_qubits)
        for j in range(Ly):
            qind_ij = qind_i + 2*j + (1 if not boundary_y[i] else 0)
            print(qind_ij)

            if i == Lx-1:
                # Last column; treatment depnds on if right boundary is smooth or rough
                if boundary_x[1] == True:
                    # right is smooth, top and bottom must be rough
                    assert boundary_y[i] == False
                    l_ind = [qind_ij, qind_ij+1, qind_ij-1]
                    l_wt = 3
                else:
                    # right is rough, top and bottom must be smooth
                    assert boundary_y[i] == True
                    if j == 0:
                        l_ind = [qind_ij, qind_ij+1, qind_ij+2*Ly-1]
                        l_wt = 3
                    elif j == Ly-1:
                        l_ind = [qind_ij, qind_ij-1, qind_ij+2*Ly-1-j]
                        l_wt = 3
                    else:
                        l_ind = [qind_ij, qind_ij+1, qind_ij-1, qind_ij+2*Ly-1-j]
                        l_wt = 4
            else:
                # Not last column
    
                # left, up, down, right
                if (j > 0 and j < Ly-1) or not boundary_y[i]:
                    # Bulk stabilizer or rough column - 4 body sta
                    l_ind = [qind_ij, qind_ij+1, qind_ij-1, qind_ij + 2*Ly-1 + (1 if not boundary_y[i] else 0) + (1 if not boundary_y[i+1] else 0)]
                    l_wt = 4
                else:
                    # Edge row for smooth boundary
                    assert boundary_y[i] == True
                    if j == 0:
                        l_ind = [qind_ij, qind_ij+1, qind_ij + 2*Ly-1 + (1 if not boundary_y[i+1] else 0)]
                        l_wt = 3
                    elif j == Ly-1:
                        l_ind = [qind_ij, qind_ij-1, qind_ij + 2*Ly-1 + (1 if not boundary_y[i+1] else 0)]
                        l_wt = 3
                    else:
                        raise ValueError()
            
            if verbose:
                print(l_ind)
    
            indices.extend(l_ind)
            indptr.append(indptr[-1]+l_wt)
    
    data = np.ones(len(indices), dtype=np.int8)
    return csr_matrix((data, indices, indptr), shape=(Lx*Ly, num_qubits), dtype=np.int8)