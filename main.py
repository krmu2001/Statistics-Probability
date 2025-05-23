import numpy as np
from Markov import udskriv, simuler, stationær_fordeling, potens, har_grænsefordeling


def main():
    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    print("\nOvergangsmatrix:")
    udskriv(P)

    print("\nSimuleret kæde (start i tilstand 0):")
    kæde = simuler(P, start=0, skridt=10)
    print(kæde)

    print("\nP^5 – sandsynligheder efter 5 skridt:")
    udskriv(potens(P, 5))

    print("\nStationær fordeling:")
    pi, entydig = stationær_fordeling(P)
    if pi is not None:
        print("π =", pi)
        print("Grænsefordeling er entydig." if entydig else "Grænsefordeling er ikke entydig.")
    else:
        print("Ingen stationær fordeling fundet.")

    print("Eksempel 1: Periodisk kæde (ingen entydig grænsefordeling)")
    P1 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    udskriv(P1)
    pi1, entydig1 = stationær_fordeling(P1)
    print("Stationær fordeling:", pi1)
    print("Entydig?", entydig1)
    print("Har grænsefordeling?", har_grænsefordeling(P1))

    print("\nEksempel 2: Aperiodisk kæde (har entydig grænsefordeling)")
    P2 = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    udskriv(P2)
    pi2, entydig2 = stationær_fordeling(P2)
    print("Stationær fordeling:", pi2)
    print("Entydig?", entydig2)
    print("Har grænsefordeling?", har_grænsefordeling(P2))

if __name__ == "__main__":
    main()