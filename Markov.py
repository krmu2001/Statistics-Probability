import numpy as np

def next_state(P, current_state):
    """
    Simulates the next state in a Markov chain.

    Parameters:
    P (np.ndarray): Transition matrix (square).
    current_state (int): Current state index.

    Returns:
    int: Next state index.
    """
    return np.random.choice(len(P), p = P[current_state])

def simulate_chain(P, initial_state, steps):
    """
    Simulates a Markov chain for a given number of steps.

    Parameters:
    P (np.ndarray): Transition matrix.
    initial_state (int): Starting state.
    steps (int): Number of steps to simulate.

    Returns:
    list of int: Sequence of state indices.
    """
    state = initial_state
    history = [state]
    for _ in range(steps):
        state = next_state(P, state)
        history.append(state)
    return history

def stationary_distribution(P, tol=1e-8, max_iter=10000):
    """
    Computes the stationary distribution of a Markov chain.

    Parameters:
    P (np.ndarray): Transition matrix.
    tol (float): Convergence tolerance.
    max_iter (int): Maximum number of iterations.

    Returns:
    np.ndarray: Stationary distribution vector.
    """
    n = len(P)
    dist = np.ones(n) / n  # Start with uniform distribution
    for _ in range(max_iter):
        new_dist = dist @ P
        if np.linalg.norm(new_dist - dist) < tol:
            return new_dist
        dist = new_dist
    raise RuntimeError("Stationary distribution did not converge.")

def print_matrix(P, precision=2):
    """
    Nicely prints a matrix with rounded values.

    Parameters:
    P (np.ndarray): Matrix to print.
    precision (int): Decimal places to round to.
    """
    print(np.array2string(P, precision=precision, floatmode='fixed'))

def matrix_power(P, n):
    """
    Computes the nth power of the transition matrix.

    Parameters:
    P (np.ndarray): Transition matrix.
    n (int): Power to raise the matrix to.

    Returns:
    np.ndarray: P^n
    """
    return np.linalg.matrix_power(P, n)
