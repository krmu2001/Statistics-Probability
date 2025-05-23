import numpy as np
from Markov import simulate_chain, stationary_distribution, print_matrix, matrix_power

def main():
    P = np.array([
        [0, 0.5, 0.5],
        [0.1, 0, 0.9],
        [0.8, 0.2, 0],
    ])

    print("Transition Matrix (P):")
    print_matrix(P)

    print("\nP^2 (Transition Probabilities After 2 Steps):")
    print_matrix(matrix_power(P, 2))

    print("\nP^5 (Transition Probabilities After 5 Steps):")
    print_matrix(matrix_power(P, 5))

    chain = simulate_chain(P, initial_state=0, steps=10)
    print("\nSimulated Markov Chain:")
    print(chain)

    stationary = stationary_distribution(P)
    print("\nStationary Distribution:")
    print(stationary)

if __name__ == "__main__":
    main()
