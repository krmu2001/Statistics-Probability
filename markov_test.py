import numpy as np
from Markov import stationær_fordeling, potens, har_grænsefordeling

def test_stationær_fordeling():
    P = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    pi, entydig = stationær_fordeling(P)
    assert entydig
    assert np.allclose(pi, [0.5, 0.5], atol=1e-6)

def test_potens():
    P = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    P2 = potens(P, 2)
    expected = np.dot(P, P)
    assert np.allclose(P2, expected, atol=1e-6)

def test_stationær_3x3():
    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    pi, entydig = stationær_fordeling(P)
    assert np.allclose(pi @ P, pi, atol=1e-6)

def test_grænsefordeling():
    P = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    assert har_grænsefordeling(P)

    P2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    assert not har_grænsefordeling(P2)


if __name__ == "__main__":
    test_stationær_fordeling()
    test_potens()
    test_stationær_3x3()
    print("Alle tests bestået.")
