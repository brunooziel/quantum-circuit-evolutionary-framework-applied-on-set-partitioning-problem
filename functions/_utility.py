from qiskit.quantum_info import SparsePauliOp
import gurobipy as gb
from gurobipy import GRB

def build_hamiltonian(parameters):
    """
    Combines the calculation of Hamiltonian coefficients and the creation of a SparsePauliOp
    in a single function for improved efficiency.
    """
    w, c, A = parameters[0], parameters[1], parameters[2]
    m = len(A)
    n = len(w)
    
    coeff = {tuple([i]): 0 for i in range(n)}
    coeff[()] = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                coeff[(i, j)] = 0
    for i in range(n):
        coeff[()] += w[i] / 2
        coeff[tuple([i])] -= w[i] / 2
    
    for i in range(m):
        temp = 0
        for k in range(n):
            for j in range(n):
                a = c[i] * A[i][j] * A[i][k] / 4
                coeff[()] += a
                coeff[tuple([j])] -= a
                coeff[tuple([k])] -= a
                if j == k:
                    coeff[()] += a
                else:
                    coeff[tuple([j, k])] += a
            coeff[()] -= A[i][k] * c[i]
            coeff[tuple([k])] += A[i][k] * c[i]
        coeff[()] += c[i]
    
    # Convert coefficients into SparsePauliOp
    pauli_terms = []
    for var, value in coeff.items():
        if value != 0.0:
            pauli_string = ['I'] * n
            for idx in var:
                pauli_string[idx] = 'Z'
            pauli_terms.append((''.join(pauli_string), value))
    
    operator_prob = SparsePauliOp.from_list(pauli_terms)
    return operator_prob

def read_instance(fname):
    f = open(fname, "r")
    lines = f.readlines()
    size = int(lines[0])
    w = [int(x) for x in lines[1].split(",")]
    c = [int(x) for x in lines[2].split(",")]
    A = []
    for i in range(3,len(lines)):
        temp = lines[i][:-1].split(" ")
        temp = [int(x) for x in temp]
        A.append(temp)
    return w,c,A

def sp_objective(w,x):
    """
    Objective function of the set partitioning problem
    :param x: binary vector of length n
    :param c: vector of length n
    :return: value of the objective function
    """
    obj = 0
    for i in range(len(w)):
        obj += w[i]*x[i]
    return obj

def sp_constraint(A,x):
    """
    Constraint function of the set partitioning problem
    :param A: matrix of size nxn
    :param x: binary vector of length n
    :return: value of the constraint function
    """
    cons = []
    for i in range(len(A)):
        temp = 0
        for j in range(len(x)):
            temp += A[i][j]*x[j]
        cons.append(temp)
    return cons

def sp_gurobi(w,A):
    """
    Classical solution using guroby for the set partitioning problem
    :param c: vector of length n
    :param A: matrix of size nxn
    :return: binary vector of length n
    """
    with gb.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gb.Model(env=env) as model:
            model.Params.LogToConsole = 0
            n = len(w)
            
            x = model.addVars(n,vtype=gb.GRB.BINARY,name='x')
            
            model.setObjective(sp_objective(w,x),GRB.MINIMIZE)
            model.addConstrs(sp_constraint(A,x)[i] == 1 for i in range(len(A)))
            model.optimize()
            return model.x
