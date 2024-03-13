import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel


def kernelLineal(X, Y):
    K = np.dot(X.T, Y)
    return K  # return the kernel matrix of X and Y


def kernelRBF(X, gammaValue):
    K = rbf_kernel(X.T, gamma=gammaValue)
    return K


def kernelPolynomial(X, Y, polyValue):
    K = polynomial_kernel(X.T, Y.T, degree=polyValue)
    return K


def kernel(X, *args, **kwargs):
    """The kernel will capture the similarity of the features."""
    Y = kwargs.get('YValue', None)
    gamma = kwargs.get('gammaValue', None)
    typeK = kwargs.get('typeK', None)
    polynomial = kwargs.get('polyValue', None)

    if (typeK == 'radial'):
        if (gamma != None):
            return kernelRBF(X, gamma)
    if (typeK == 'linear'):
        if (Y is not None):
            return kernelLineal(Y, X)
    if (typeK == 'polynomial'):
        return kernelPolynomial(X, Y, polynomial)
    return None


# Parametros typeK, gammas, rank
def SLMVPTrain(X, Y, rank, typeK, gammaX, gammaY, polyValue, multilabel=None, debug=False):

    # Performs Singular value decomposition
    Ux, sx, Vx = np.linalg.svd(X, full_matrices=False)

    # Put the singular values sx in the diagonal of a zero matrix
    Sx = np.zeros((sx.shape[0], sx.shape[0]))
    Sx[:sx.shape[0], :sx.shape[0]] = np.diag(sx)

    KXX = kernel(X, typeK=typeK, YValue=X,
                 gammaValue=gammaX, polyValue=polyValue)
    # KXX = kernel(Y, typeK='lineal',YValue = X)

    # Centering KXX
    l = KXX.shape[0]
    j = np.ones(l)

    # Quitar la media de KXX
    KXX = KXX - (np.dot(np.dot(j, j.T), KXX))/l - (np.dot(KXX, np.dot(j, j.T))) / \
        l + (np.dot((np.dot(j.T, np.dot(KXX, j))), np.dot(j, j.T)))/(np.power(l, 2))

    if multilabel is None:
        Y = np.reshape(Y, (1, Y.size))
    else:
        Y = Y.T

    KYY = kernel(Y, typeK=typeK, YValue=Y,
                 gammaValue=gammaY, polyValue=polyValue)

    # KYY = kernel(Y, typeK='lineal',YValue = Y)

    # Centering KYY
    KYY = KYY - (np.dot(np.dot(j, j.T), KYY))/l - (np.dot(KYY, np.dot(j, j.T))) / \
        l + (np.dot((np.dot(j.T, np.dot(KYY, j))), np.dot(j, j.T)))/(np.power(l, 2))

    # Joint similarity matrix
    KXXKYY = np.dot(KXX, KYY)
    KXXKYYR = KXXKYY

    # Poner rank al max 100, 1000
    KXXKYYR = np.dot(np.dot(Vx[:, :], KXXKYYR), Vx[:, :].T)

    # Obtaining the linear embedding B
    Ub, Sb, Vb = np.linalg.svd((KXXKYYR), full_matrices=False)
    Sx = Sx[:, :]
    B = np.dot(np.dot(Ux[:, :], np.linalg.inv(Sx)), Ub)

    # Projections on the learned space
    # P = np.dot(B.T,X)

    # XKKXKKYX = np.dot(np.dot(X, KXXKYY), X.T)
    # U, s, V = np.linalg.svd(XKKXKKYX)

    # return U

    if debug == False:
        return B[:, 1:rank+1]  # return the learned model
    else:
        return KXXKYY, KXXKYYR, B, X


def SLMVP_transform(B, X):
    P = np.dot(B, X)
    return P
