import numpy as np

# reference Python for Finance p.295
# reference Hull ch.13
# reference Hull ch.21

# option payoff function
def payoff_function(option_type, S, K, 
                    alternative = 0):
    
    if option_type == "Call": 
        return np.maximum(S - K, alternative)
    
    elif option_type == "Put": 
        return np.maximum(K - S, alternative)
    
    else:
        raise ValueError("'Call' or 'Put'" )

# creat 2 empty lists to store price trees for Greeks calculation
STK = []

def binomial_tree(S, K, T, r, sigma, q, N, 
                  option_style, option_type):

    ######################## This is part one ########################

    # references Python for Finance p.295
    
    # dt, u, d, a, p, and discount factor
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt)) 
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # ndarray object with gross upward movements
    _ = np.arange(N + 1)
    up = np.resize(_, (N + 1, N + 1))

    #  ndarray object with gross downward movements
    down = up.T * 2

    ST = S * np.exp(sigma * np.sqrt(dt) * (up - down))

    # append S tree to STK for Greeks calculation
    STK.append(ST)

    # get the last node and pass it to the recursive function
    last_node = payoff_function(option_type, ST[:,N], K)


    ######################## This is part two - Recursive Function ########################


    def binomial_recursive(payoff):
        
        if option_style == "European":

            pu = payoff[:-1]
            pd = payoff[1:]
            price = ((pu * p) + (pd * (1 - p))) * discount

            # store the first 3 nodes
            if price.shape[0] < 4:
                for_greeks.append(price)

            # return the price
            if price.shape[0] == 1:
                return price
            
        elif option_style == "American":
            
            pu = payoff[:-1]
            pd = payoff[1:]
            price = ((pu * p) + (pd * (1 - p))) * discount

            # generalize the col
            col = -(N - price.shape[0] + 2)


            stock_price = ST[:price.shape[0], col]

            # compare intrinsic value
            price = payoff_function(option_type, 
                                    S = stock_price, 
                                    K = K, alternative = price)

            # store the first 3 nodes
            if price.shape[0] < 4:
                for_greeks.append(price)
            
            # return the price
            if price.shape[0] == 1:
                return price            

        else:
            raise ValueError("'Call' or 'Put'" )
            
            
        return binomial_recursive(price)   
    
    # pass the last node to the recursive function
    output = binomial_recursive(last_node)

    return float(output)