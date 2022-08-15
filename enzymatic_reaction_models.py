# Enzymatic reaction system
# Model 1 (m1): Full model
# Model 2 (m2): Model with exact conservation laws
# Model 3 (m3): Model with assumed conservation laws
# Model 4 (m4): Reduced model with quasi-steady state approximation (QSSA)

import numpy as np

def m1(t, x, *args):
    """Full (scipy.integrate.odeint solveable) model 
       of enzymatic reaction system:
        E + S <-> C, forward rate = a, reverse rate = d
        C --> E + P, forward rate = k
        Using mass-action kinetics, we write an ODE model.
    Args:
        t (float): Time point at which the RHS of ODE is computed
        x (numpy.ndarray): Numpy array for model state variables
        args (tuple): Model parameters passed in as args
    Returns:
        A numpy.ndarray for dx/dt, where x is the state vector
    """
    E, S, C, P = x
    a, d, k = args
    dE_dt = -a*E*S + (d+k)*C
    dS_dt = -a*E*S + d*C
    dC_dt = a*E*S - (d+k)*C
    dP_dt = k*C
    return np.array([dE_dt, dS_dt, dC_dt, dP_dt])

def m2(t, x, *args):
    """Mass conservation (scipy.integrate.odeint solveable) model 
       of enzymatic reaction system:
       Using laws of conservation, we write a reduced ODE model:
       E + C = E_tot
       S + C + P = S_tot
    Args:
        t (float): Time point at which the RHS of ODE is computed
        x (numpy.ndarray): Numpy array for model state variables
        args (tuple): Model parameters passed in as args
    Returns:
        A numpy.ndarray for dx/dt, where x is the state vector
    """
    C, P = x
    a, d, k, E_tot, S_tot = args
    Kd = d/a
    eps = k/d
    dC_dt = (1/eps)*(k/Kd * (E_tot - C) * (S_tot - C - P) - k*C) - k*C
    dP_dt = k*C
    return np.array([dC_dt, dP_dt])

def m3(t, x, *args):
    """Mass conservation (scipy.integrate.odeint solveable) model 
       of enzymatic reaction system:
       Using laws of conservation, we write a reduced ODE model:
       E + C = E_tot
       S + C + P = S_tot
       then, under further assumption that S_tot >> E_tot, we have
       C << S_tot, so we simplify the model further.
    Args:
        t (float): Time point at which the RHS of ODE is computed
        x (numpy.ndarray): Numpy array for model state variables
        args (tuple): Model parameters passed in as args
    Returns:
        A numpy.ndarray for dx/dt, where x is the state vector
    """
    C, P = x
    a, d, k, E_tot, S_tot = args
    Kd = d/a
    eps = k/d
    dC_dt = (1/eps)*(k/Kd * (E_tot - C) * (S_tot - P) - k*C) - k*C
    dP_dt = k*C
    return np.array([dC_dt, dP_dt])

def m4(t, x, *args):
    """QSSA (scipy.integrate.odeint solveable) model 
    of enzymatic reaction system:
    Using quasi-steady state approximation, we get a one-dimensional model
    Args:
        t (float): Time point at which the RHS of ODE is computed
        x (numpy.ndarray): Numpy array for model state variables
        args (tuple): Model parameters passed in as args
    Returns:
        A numpy.ndarray for dx/dt, where x is the state vector
    """
    P = x
    a, d, k, E_tot, S_tot = args
    Km = (d + k)/a
    dP_dt = k*E_tot*(S_tot - P)/(S_tot - P + Km)
    return np.array(dP_dt)

# Parameter values
a = 10
d = 10
k = 0.1
S_tot = 100
E_tot = 1

# The condition vector for hierarchical validation
phi = np.array([d, S_tot])

# Compute goodness metrics for models given phi against
# the ground truth model m1
from scipy.integrate import odeint
from scipy.linalg import norm

timepoints = np.linspace(0, 1500, 100)
shape_t = np.shape(timepoints)[0]
ground_truth = {'E':np.zeros(shape_t),
                'S':np.zeros(shape_t),
                'C':np.zeros(shape_t),
                'P':np.zeros(shape_t)}
params = (a, d, k)
init_cond = np.array([E_tot, S_tot, 0, 0])
sol = odeint(m1, t=timepoints, y0=init_cond, args=params, tfirst=True)
ground_truth['E'] = sol[:,0]
ground_truth['S'] = sol[:,1]
ground_truth['C'] = sol[:,2]
ground_truth['P'] = sol[:,3]

def g_m2(phi, timepoints):
    """Returns goodness metric (float) by computing summed norm 
    of difference between states of the model m2 and ground truth m1
    at the given condition on parameters: phi.

    Args:
        phi (np.ndarray): Input condition vector
    """
    d, S_tot = phi
    params = (a, d, k, E_tot, S_tot)
    shape_t = np.shape(timepoints)[0] 
    states_m2 = ['C', 'P']
    init_cond = np.zeros(len(states_m2))
    sol = odeint(m2, t=timepoints, y0=init_cond, args=params, tfirst=True)
    sol_m2 = {'C':np.zeros(shape_t),
              'P':np.zeros(shape_t)}
    sol_m2['C'] = sol[:,0]
    sol_m2['P'] = sol[:,1]
    error = 0
    for state in states_m2:
        error += norm(sol_m2[state] - ground_truth[state], ord=2)
    return -1*error
    
def g_m3(phi, timepoints):
    """Returns goodness metric (float) by computing summed norm 
    of difference between states of the model m3 and ground truth m1
    at the given condition on parameters: phi.

    Args:
        phi (np.ndarray): Input condition vector
    """
    d, S_tot = phi
    params = (a, d, k, E_tot, S_tot)
    shape_t = np.shape(timepoints)[0] 
    states_m3 = ['C', 'P']
    init_cond = np.zeros(len(states_m3))
    sol = odeint(m3, t=timepoints, y0=init_cond, args=params, tfirst=True)
    sol_m3 = {'C':np.zeros(shape_t),
              'P':np.zeros(shape_t)}
    sol_m3['C'] = sol[:,0]
    sol_m3['P'] = sol[:,1]
    error = 0
    for state in states_m3:
        error += norm(sol_m3[state] - ground_truth[state], ord=2)
    return -1*error

def g_m4(phi, timepoints):
    """Returns goodness metric (float) by computing summed norm 
    of difference between states of the model m4 and ground truth m1
    at the given condition on parameters: phi.

    Args:
        phi (np.ndarray): Input condition vector
    """
    d, S_tot = phi
    params = (a, d, k, E_tot, S_tot)
    shape_t = np.shape(timepoints)[0] 
    states_m4 = ['P']
    init_cond = np.zeros(len(states_m4))
    sol = odeint(m4, t=timepoints, y0=init_cond, args=params, tfirst=True)
    sol_m4 = {'P':np.zeros(shape_t)}
    sol_m4['P'] = sol[:,0]
    error = 0
    for state in states_m4:
        error += norm(sol_m4[state] - ground_truth[state], ord=2)
    return -1*error

all_phi = [
    np.array([10, 100]),
    np.array([1, 100]),
    np.array([10, 1]),
    np.array([1, 1])
]
for phi_i in all_phi:
    print("For condition = {0}, the goodness of m2 = {1}".format(phi_i, g_m2(phi_i, timepoints)))
    print("For condition = {0}, the goodness of m3 = {1}".format(phi_i, g_m3(phi_i, timepoints)))
    print("For condition = {0}, the goodness m4 = {1}".format(phi_i, g_m4(phi_i, timepoints)))