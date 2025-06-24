import numpy as np
import ants

__all__ = ['polar_decomposition']

def polar_decomposition(X):
    """Perform a polar decomposition of a matrix.

    The polar decomposition of a matrix X is a factorization of the form:

    X = P @ Z

    where P is an orthogonal projection matrix and Z is an orthogonal matrix.

    Arguments
    ---------
    X : numpy.ndarray
        Input matrix to decompose. Should be a square matrix.

    Returns
    -------
    dict
        A dictionary containing:
        - "P": the orthogonal projection matrix.
        - "Z": the orthogonal part of the decomposition.
        - "Xtilde": the reconstructed matrix from the decomposition (Xtilde = P @ Z).
    """
    U, d, Vh = np.linalg.svd(X, full_matrices=False) # Vh is V transpose
    # This is the formula for P in a LEFT decomposition (X = PZ)
    P = np.matmul(U, np.matmul(np.diag(d), np.transpose(U)))
    # Z is the orthogonal part, U @ Vh
    Z = np.matmul(U, Vh)

    # Correction to ensure Z is a proper rotation (det(Z) = +1)
    if np.linalg.det(Z) < 0:
        n = X.shape[0]
        reflection_matrix = np.identity(n)
        reflection_matrix[-1, -1] = -1.0 # More robust to change last element
        U_prime = U.copy()
        U_prime[:, -1] *= -1
        Z = U_prime @ Vh

    # The returned Xtilde is P @ Z, consistent with a LEFT decomposition
    return({"P" : P, "Z" : Z, "Xtilde" : np.matmul(P, Z)})