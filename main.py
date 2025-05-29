import numpy as np
from Markov import (
    udskriv, potens, stationær_fordeling, har_grænsefordeling,
    mean_hitting_times, vis_fordeling, simuler_kæde, p_n_skridt,
    er_irreducibel, er_aperiodisk, er_absorberende,
    absorptions_sandsynligheder, alle_rekurrente_og_transiente
)

def main():
    print("EKSEMPEL: Markov-kæde med 3 tilstande (periodisk)\n")
    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    udskriv(P)

    print("\nSimuler kæde fra tilstand 0:")
    print(simuler_kæde(P, start=0, skridt=10))

    print("\nP^5 (5 skridt frem):")
    udskriv(potens(P, 5))

    print("\nStationær fordeling:")
    pi, entydig = stationær_fordeling(P)
    vis_fordeling(pi)
    print("Entydig grænsefordeling?" , "Ja" if entydig else "Nej")

    print("\nHar kæden grænsefordeling?")
    print("Ja" if har_grænsefordeling(P) else "Nej")

    print("\nForventet tid til at ramme tilstand 2:")
    h = mean_hitting_times(P, target_state=2)
    print(h)

    print("\nSandsynlighed for at gå fra 0 → 2 på 2 skridt:")
    print(p_n_skridt(P, start=0, slut=2, n=2))

    print("\nEr kæden irreducibel?")
    print("Ja" if er_irreducibel(P) else "Nej")

    print("\nEr kæden aperiodisk?")
    print("Ja" if er_aperiodisk(P) else "Nej")

    print("\nIndeholder kæden absorberende tilstande?")
    print("Ja" if er_absorberende(P) else "Nej")

    print("\nRekurrente og transiente tilstande:")
    rek, trans = alle_rekurrente_og_transiente(P)
    print("Rekurrente:", rek)
    print("Transiente:", trans)

    print("\nAbsorptionssandsynligheder (kun hvis relevant):")
    # Kun relevant hvis kæden har absorberende tilstande – her er et eksempel:
    P_abs = np.array([
        [0.5, 0.5, 0.0],
        [0.2, 0.3, 0.5],
        [0.0, 0.0, 1.0]  # absorberende tilstand
    ])
    print("Ny kæde:")
    udskriv(P_abs)
    B = absorptions_sandsynligheder(P_abs, absorberende_tilstande=[2])
    print("Absorptionssandsynligheder fra transiente:")
    print(B)

if __name__ == "__main__":
    main()
