import numpy as np

def mc(S, K, r, sigma, T, steps, N, option_type):
    dt = T / steps
    epsilon = np.random.normal(size = (steps, N))
    ST = np.log(S) + np.cumsum((r - 0.5 * sigma**(2)) * dt + sigma  * epsilon * np.sqrt(dt), axis = 0)
    ST = np.exp(ST)

    if option_type == "call":
        CT = np.maximum(ST[-1] - K, 0).mean()
        price = CT * np.exp(-r * T)
        return price
    
    elif option_type == "put":
        PT = np.maximum(K - ST[-1], 0).mean()
        price = PT * np.exp(-r * T)
        return price

    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
