import numpy as np

from switss.solver import LP
from switss.model import AbstractMDP


def check_mec_certificate(amdp : AbstractMDP, mecs, mec_certificate, tol=1e-6):
    """
    """
    mec_quotient_ec_free_cert, _ = mec_certificate

    # get mec quotient mdp
    q_mdp, original_inner_action_codes = amdp.mec_quotient_mdp(mecs)
    # get system matrix of the mec quotient mdp
    q_A = q_mdp.get_system_matrix()
    # multiply the system matrix with mec_quotient_ec_free_certificate
    M = np.matmul(q_A, np.matrix(mec_quotient_ec_free_cert).transpose())
    # check if the multiplication results fulfils >= 1 for each constraint
    for i in range(M.shape[0]):
        if M[(i, 0)] + tol < 1:
            return False

    #

    return True

def generate_mec_certificate(amdp : AbstractMDP, mecs, certificate_bounds=1e9):
    """
    """
    # get mec quotient mdp
    q_mdp, original_inner_action_codes = amdp.mec_quotient_mdp(mecs)
    # get system matrix of the mec quotient mdp
    q_A = q_mdp.get_system_matrix()
    # constraints count and variables count in the system matrix   
    constraints = q_A.shape[0]
    variables = q_A.shape[1]
    # create matrix for LP from the system matrix and add constrainst for bounding the variables
    A = np.concatenate((q_A, np.zeros(shape=(2*variables, variables))), axis=0)
    b = np.array(constraints*[1] + 2*variables*[-certificate_bounds])
    opt = np.array(variables*[1])
    for i in range(variables):
        A[(constraints + 2*i, i)] = 1
        A[(constraints + 2*i + 1, i)] = -1
    # calculate the variables in the certificate for EC=freeness of the quotient mdp
    lp = LP.from_coefficients(A,b,opt,sense=">=")
    result = lp.solve(solver="gurobi")
    mec_quotient_ec_free_cert = result.result_vector

    #


    mec_certificate = (mec_quotient_ec_free_cert, [])
    assert check_mec_certificate(amdp, mecs, mec_certificate)
    return mec_certificate

