import numpy as np

def next_step(P, s):
    """
    Returnér næste tilstand givet nuværende tilstand s og overgangsmatrix P.

    Parametre:
    - P: overgangsmatrix (numpy array)
    - s: nuværende tilstand (heltal)

    Returnerer:
    - næste tilstand (heltal)
    """
    return np.random.choice(len(P), p=P[s])

def simuler(P, start, skridt):
    """
    Simuler en Markov-kæde.

    Parametre:
    - P: overgangsmatrix
    - start: starttilstand (indeks)
    - skridt: antal skridt i kæden

    Returnerer:
    - liste med tilstande
    """
    hist = [start]
    for _ in range(skridt):
        hist.append(next_step(P, hist[-1]))
    return hist

def stationær_fordeling(P):
    """
    Finder stationær fordeling π som opfylder πP = π og summerer til 1.

    Returnerer:
    - π: numpy array med sandsynligheder
    - entydig: bool – om fordelingen er entydig (grænsefordeling eksisterer)
    """
    n = P.shape[0]
    A = np.transpose(P) - np.eye(n)
    A = np.vstack([A, np.ones(n)])  # π1 + π2 + ... + πn = 1
    b = np.zeros(n + 1)
    b[-1] = 1

    try:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, False

    if np.all(pi >= -1e-10):  # tillad lidt negativt pga. numeriske fejl
        pi = np.clip(pi, 0, 1)
        pi /= np.sum(pi)
        return pi, True

    return None, False

def har_grænsefordeling(P, n=100, tolerance=1e-6):
    """
    Undersøger om Markovkæden har en entydig grænsefordeling.

    Parametre:
    - P: overgangsmatrix
    - n: antal iterationer (hvor langt vi går frem i tiden)
    - tolerance: hvor ens rækkerne i P^n skal være

    Returnerer:
    - True hvis alle rækker i P^n er næsten ens
    - False ellers
    """
    Pn = np.linalg.matrix_power(P, n)
    første_række = Pn[0]
    for i in range(1, Pn.shape[0]):
        if not np.allclose(Pn[i], første_række, atol=tolerance):
            return False
    return True

def udskriv(P, decimaler=2):
    """
    Udskriv en matrix pænt afrundet.

    Parametre:
    - P: matrix (numpy array)
    - decimaler: antal decimaler
    """
    print(np.array2string(P, precision=decimaler, floatmode='fixed'))

def potens(P, n):
    """
    Udregn P^n – overgangsmatrix efter n skridt.

    Parametre:
    - P: overgangsmatrix
    - n: antal skridt

    Returnerer:
    - numpy array med P^n
    """
    return np.linalg.matrix_power(P, n)
