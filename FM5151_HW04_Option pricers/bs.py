import numpy as np
from math import e
from scipy.stats import norm

def black_scholes_dividend(S, K, r, q, sigma, T, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5*sigma**(2)) * T) / (sigma * T**(0.5))
    d2 = d1 - (sigma * T**(0.5))
    
    if option_type == "call":
        return (S * e**(-q * T)) * norm.cdf(d1) - K * e**(-r * T) * norm.cdf(d2)
    
    elif option_type == "put":
        return K * e**(-r * T) * norm.cdf(-d2) - (S * e**(-q * T)) * norm.cdf(-d1)
    
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")