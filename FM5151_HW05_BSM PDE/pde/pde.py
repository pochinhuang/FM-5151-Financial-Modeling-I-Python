import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def pde(S, r, q, sigma, M, N, T, K, otype, is_am):

    delta_t = T / N #time
    delta_s = S / M #price
    
    s_values = np.linspace(0, S, M)
    t_values = np.linspace(0, T, N)

    f = np.zeros((M , N))
    E = np.zeros(M - 2)


    if otype == "put":
        f[:, -1] = np.maximum(K - S, 0)
        f[-1, :] = 0
        f[0, :] = K
    elif otype == "call":
        f[:, -1] = np.maximum(S - K, 0)
        f[-1, :] = S
        f[0, :] = 0
    else:
        raise ValueError("call or put")


    aj = ((1 / 2) * (r - q) * delta_s * delta_t) - ((1 / 2) * sigma**(2) * delta_s**(2) * delta_t)
    bj = 1 + (sigma**(2) * delta_s**(2) * delta_t) + (r * delta_t)
    cj = ((-1 / 2) * (r - q) * delta_s * delta_t) - ((1 / 2) * sigma**(2) * delta_s**(2) * delta_t)

    abc_matrix = sparse.diags([aj, bj, cj], [-1, 0, 1], shape = (M - 2, M - 2)).tocsc()

    for i in range(N - 2, -1, -1):
        E[0] = aj * f[0, i]
        E[-1] = cj * f[-1, i]
        f[1:-1, i] = spsolve(abc_matrix, (f[1:-1, i + 1] - E))

        if is_am == "True":
            if otype == "put":
                f[1:-1, i] = np.maximum(f[1:-1, i], np.maximum(K - s_values[1:-1], 0))
            elif otype == "call":
                f[1:-1, i] = np.maximum(f[1:-1, i], np.maximum(s_values[1:-1] - K, 0))
            else:
                raise ValueError("call or put")
         
    price = np.interp(S, s_values, f[:, 0])   
    
    return s_values, t_values, f, price