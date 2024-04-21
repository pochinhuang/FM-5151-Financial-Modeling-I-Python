import numpy as np
from math import e
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, r, q, sigma, T, iscall):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.iscall = iscall

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.iscall:
            price = (self.S * e**(-self.q * self.T)) * norm.cdf(d1) - self.K * e**(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * e**(-self.r * self.T) * norm.cdf(-d2) - (self.S * e**(-self.q * self.T)) * norm.cdf(-d1)

        return price

    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.iscall:
            delta = norm.cdf(d1) * e**(-self.q * self.T)
        else:
            delta = -norm.cdf(-d1) * e**(-self.q * self.T)

        return delta

    def gamma(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        gamma = norm.pdf(d1) * e**(-self.q * self.T) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma

    def theta(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        theta = -(self.S * norm.pdf(d1) * self.sigma * e**(-self.q * self.T)) / (2 * np.sqrt(self.T))

        if self.iscall:
            theta -= self.r * self.K * e**(-self.r * self.T) * norm.cdf(d1)
        else:
            theta += self.r * self.K * e**(-self.r * self.T) * norm.cdf(-d1)

        return theta

    def vega(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T) * e**(-self.q * self.T)
        return vega

    def rho(self):
        d2 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T)) - self.sigma * np.sqrt(self.T)

        if self.iscall:
            rho = self.K * self.T * e**(-self.r * self.T) * norm.cdf(d2)
        else:
            rho = -self.K * self.T * e**(-self.r * self.T) * norm.cdf(-d2)

        return rho

