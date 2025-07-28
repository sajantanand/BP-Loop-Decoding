import numpy as np
from codes import toric


def fixed_X_error(L, k, synd):
    """
    We have measured the Z stabiliers to get the syndrome. We want to choose a valid "recovery string" of X
    operators that will creates the observed syndrome and thus returns the code to its codespace. We do this
    by dragging all m (magnetic flux, plaquette) anyons to the smooth boundary at the top left.

    We need to return the applied string as this is important for determining the logical class. 
    """
    Lx, Ly, boundary_x, boundary_y, seen_qubits, num_qubits = surface_code_properties(L, k, verbose)



def fixed_Z_error(L, k, synd):
    """
    Do the same as `fixed_X_error`, but for X stabilizer measurements. We are choosing a Z error string by
    dragging all e anyons to the left rough boundary.
    """
