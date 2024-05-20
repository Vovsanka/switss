import numpy as np

from switss.solver import LP
from switss.model import AbstractMDP
from collections import deque


def check_mec_certificate(amdp : AbstractMDP, mecs, mec_certificate, tol=1e-6):
    """
    """
    mec_quotient_ec_free_cert, mec_strongly_connected_cert = mec_certificate
    
    ### check the certificate for mec quotient ec-freeness
    # get mec quotient mdp
    q_mdp, original_inner_action_codes = amdp.mec_quotient_mdp(mecs)
    # get system matrix of the mec quotient mdp
    q_A = q_mdp.get_system_matrix()
    # multiply the system matrix with mec_quotient_ec_free_certificate
    try:
        M = np.matmul(q_A, np.matrix(mec_quotient_ec_free_cert).transpose())
    except:
        return False
    # check if the multiplication results fulfils >= 1 for each constraint
    for i in range(M.shape[0]):
        if M[(i, 0)] + tol < 1:
            return False

    ### check the certificate for mec being strongly connected
    vertex_count = amdp.P.shape[1]
    # retrieve fwd and bwd from certificate
    fwd, bwd = mec_strongly_connected_cert
    # check fwd, bwd are functions: S -> N
    if fwd.shape[0] != vertex_count or bwd.shape[0] != vertex_count:
        return False
    if not np.issubdtype(fwd.dtype, np.integer) or not np.issubdtype(bwd.dtype, np.integer):
        return False
    # the fwd and the bwd constraints are fulfiled by vertex
    fwd_ok = np.zeros(vertex_count, dtype=bool)
    bwd_ok = np.zeros(vertex_count, dtype=bool)
    # check each inner action
    components,_,mec_counter = mecs
    for code, v_state in list(amdp.P.keys()):
        if code not in original_inner_action_codes:
            continue
        u_state, action = amdp.index_by_state_action.inv[code]
        if fwd[v_state] < fwd[u_state]:
            fwd_ok[u_state] = True
        if bwd[u_state] < bwd[v_state]:
            bwd_ok[v_state] = True
    # all elements except of component leaders (bwd[u] = fwd[u] = 0) must be True
    if not np.all(fwd_ok == bwd_ok):  
        return False
    # check leaders-components bijection (each component has exactly one leader)
    component_leaders = np.zeros(mec_counter, dtype=int)
    for state,ok in enumerate(fwd_ok):
        if not ok:  # then state is a leader
            component_leaders[components[state]] += 1
    if not (component_leaders == 1).all():
        return False

    return True

def generate_mec_certificate(amdp : AbstractMDP, mecs, certificate_bounds=1e9):
    """
    """
    ### generate the certificate for mec quotient ec-freeness
    # get mec quotient mdp
    q_mdp, original_inner_action_codes = amdp.mec_quotient_mdp(mecs)
    # get system matrix of the mec quotient mdp
    q_A = q_mdp.get_system_matrix()
    # constraints count and variables count in the system matrix   
    constraints = q_A.shape[0]
    variables = q_A.shape[1]
    # create matrix for LP from the system matrix and add constrainst for bounding the variables
    A = q_A
    b = np.array(list(constraints*[1]))
    opt = np.array(variables*[0])
    # calculate the variables in the certificate for EC=freeness of the quotient mdp
    lp = LP.from_coefficients(A,b,opt,sense=">=")
    result = lp.solve(solver="gurobi")
    # provide certificate
    mec_quotient_ec_free_cert = result.result_vector

    ### generate the certificate for mec being strongly connected
    vertex_count = amdp.P.shape[1]
    # init adjacency lists
    adj = {}  # adjacency list, contains only edges from inner actions
    rev_adj = {} # reversed adjacency list
    for u in range(vertex_count):
        adj[u] = []
        rev_adj[u] = []
    # calculate the ajacency lists only from inner actions
    original_inner_action_codes = set(original_inner_action_codes)
    components,_,_ = mecs
    for code, v_state in list(amdp.P.keys()):
        if code not in original_inner_action_codes:
            continue
        u_state, action = amdp.index_by_state_action.inv[code]
        adj[u_state].append(v_state)
        rev_adj[v_state].append(u_state)
    # init the certifying functions fwd, bwd
    fwd = np.zeros(vertex_count, dtype=int)
    bwd = np.zeros(vertex_count, dtype=int)
    # processed vertices have once been visited in bfs
    processed = np.zeros(vertex_count, dtype=bool)
    # breadth first seach with arbitrary adjacency list and callback function vertices
    def bfs(adjacent, source, cb):
        visited = np.zeros(vertex_count, dtype=bool)
        distance = np.zeros(vertex_count, dtype=int)
        queue = deque([source])
        visited[source] = processed[source] = True
        while len(queue) > 0:
            current_vertex = queue.popleft()
            cb(current_vertex, distance[current_vertex])
            for next_vertex in adjacent[current_vertex]:
                if visited[next_vertex]:
                    continue
                visited[next_vertex] = processed[next_vertex] = True
                distance[next_vertex] = distance[current_vertex] + 1
                queue.append(next_vertex)
    # functions to be called in bfs
    def set_fwd(vertex, dist):
        fwd[vertex] = dist
    def set_bwd(vertex, dist):
        bwd[vertex] = dist
    # process all vertices with bfs and calculate fwd and bwd
    for u in range(vertex_count):
        if processed[u]:
            continue
        bfs(adj, u, set_bwd)
        bfs(rev_adj, u, set_fwd)
    # provide certificate
    mec_strongly_connected_cert = (fwd, bwd)

    # unite certificates
    mec_certificate = (mec_quotient_ec_free_cert, mec_strongly_connected_cert)
    assert check_mec_certificate(amdp, mecs, mec_certificate)
    return mec_certificate

