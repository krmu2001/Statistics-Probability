import numpy as np

def næste_tilstand(P, s):
    return np.random.choice(len(P), p=P[s])

def simuler_kæde(P, start, skridt):
    hist = [start]
    for _ in range(skridt):
        hist.append(næste_tilstand(P, hist[-1]))
    return hist

def udskriv(P, decimaler=2):
    print(np.array2string(P, precision=decimaler, floatmode='fixed'))

def potens(P, n):
    return np.linalg.matrix_power(P, n)

def stationær_fordeling(P):
    n = P.shape[0]
    A = np.transpose(P) - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1

    try:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.all(pi >= -1e-10):
            pi = np.clip(pi, 0, 1)
            pi /= np.sum(pi)
            return pi, True
    except:
        pass
    return None, False

def har_grænsefordeling(P, n=100, tolerance=1e-6):
    Pn = np.linalg.matrix_power(P, n)
    første = Pn[0]
    for i in range(1, len(Pn)):
        if not np.allclose(Pn[i], første, atol=tolerance):
            return False
    return True

def vis_fordeling(pi, labels=None):
    for i, val in enumerate(pi):
        navn = f"Tilstand {i}" if not labels else labels[i]
        print(f"{navn}: {val:.4f}")

def p_n_skridt(P, start, slut, n):
    return potens(P, n)[start, slut]

def mean_hitting_times(P, target_state):
    n = P.shape[0]
    indices = [i for i in range(n) if i != target_state]
    A = np.eye(len(indices)) - P[np.ix_(indices, indices)]
    b = np.ones(len(indices))
    h = np.linalg.solve(A, b)
    full_h = np.zeros(n)
    for idx, i in enumerate(indices):
        full_h[i] = h[idx]
    return full_h

def er_irreducibel(P):
    n = len(P)
    reachable = np.linalg.matrix_power(P + np.eye(n), n)
    return np.all(reachable > 0)

def er_aperiodisk(P):
    n = len(P)
    g = np.zeros_like(P)
    for k in range(1, n * n):
        g += (potens(P, k) > 0).astype(int)
    return np.all(g > 0)

def er_absorberende(P):
    return any(np.all(P[i] == np.eye(len(P))[i]) for i in range(len(P)))

def absorptions_sandsynligheder(P, absorberende_tilstande):
    n = len(P)
    transienter = [i for i in range(n) if i not in absorberende_tilstande]
    Q = P[np.ix_(transienter, transienter)]
    R = P[np.ix_(transienter, absorberende_tilstande)]
    I = np.eye(len(Q))
    N = np.linalg.inv(I - Q)
    B = N @ R
    return B

def alle_rekurrente_og_transiente(P):
    n = len(P)
    R = []
    T = []
    for i in range(n):
        prob_sum = 0
        Pi = np.linalg.matrix_power(P, 1)
        for k in range(1, 100):
            prob_sum += Pi[i, i]
            Pi = Pi @ P
        if prob_sum > 0.999:
            R.append(i)
        else:
            T.append(i)
    return R, T
