import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import random

seed = random.SystemRandom().randint(1, 999999)

np.random.seed(seed)

# Silence progress in cvxopt solver
solvers.options['show_progress'] = False

# Number of assets
NUM_ASSETS = 10

# Number of observations of asset returns
NUM_OBSERVATIONS = 1000

# Number of sample portfolios to plot
NUM_PORTFOLIOS = 500

# Capital
CAPITAL = 10**6

## Produces a list of weights (percentage of capital invested in each asset)
#  must sum to 1 given all capital is invested
def rand_weights(n):
    weights = np.random.rand(n)
    return weights / sum(weights)

## Returns mean and standard deviation (estimate of volatility) of random portfolio
def random_portfolio(returns):

    # Vector of means of returns
    R = np.asmatrix(np.mean(returns, axis=1))

    # Portfolio weights vector
    # Shape returns (NUM_ASSETS, NUM_OBSERVATIONS)
    # i.e. shape of matrix, n x m
    w = np.asmatrix(rand_weights(returns.shape[0]))

    # Returns covariance matrix
    C = np.asmatrix(np.cov(returns))

    # From https://en.wikipedia.org/wiki/Modern_portfolio_theory
    # mu is expected return of portfolio
    # sigma is S.D. (risk) of portfolio
    mu = w * R.T
    sigma = np.sqrt(w * C * w.T)

    # Filter large SDs
    # removes portfolios with crazy risk
    if sigma > 2:
        return random_portfolio(returns)

    return mu, sigma

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))

    # np.mean computes vector of means along
    # as mean of each row
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    # -- documentation at http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    G = -opt.matrix(np.eye(n)) # negative nxn id
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    ## Calculate efficient frontier weights using quadratic programming
    ## https://en.wikipedia.org/wiki/Quadratic_programming
    # Minimizes (1/2) w^T * (mu * S) * w - pbar^T * w for weights w
    # subject to the constrains:
    # G * w <= h (component-wise) -- positive weights
    # and A*x = b -- Sum of weights == 1
    # where S is the covariance matrix,
    # pbar are the mean returns,
    # mu is one of N sample means -- implying parametric solution
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    ## Could replace above with Lagrange multiplier (or convex hull)

    # Calculate risks and returns for frontier
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Calculate quadratic for the frontier curve
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # Calculate the optimal portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks


return_vec = np.random.randn(NUM_ASSETS, NUM_OBSERVATIONS)
means, stds = np.column_stack([random_portfolio(return_vec) for _ in xrange(NUM_PORTFOLIOS)])
weights, returns, risks = optimal_portfolio(return_vec)

print weights
print sum(weights)
print weights*CAPITAL

plt.plot(stds, means, 'o', markersize=5)
plt.plot(risks, returns, 'y-o')
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()