import numpy as np
# import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import random
import pandas as pd
from zipline.utils.factory import load_bars_from_yahoo
from zipline import TradingAlgorithm
from zipline.api import (history,
                         set_slippage,
                         slippage,
                         set_commission,
                         commission,
                         order_target_percent)
from zipline.api import add_history
end = pd.Timestamp.utcnow()
start = end - 2500 * pd.tseries.offsets.BDay()

data = load_bars_from_yahoo(stocks=['IBM', 'GLD', 'XOM', 'AAPL',
                                    'MSFT', 'GOOG', 'SHY'],
                            start=start, end=end)

seed = random.SystemRandom().randint(1, 999999)

np.random.seed(123)

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
    qs = [10**(-5.0 * t/N + 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))

    # np.mean computes vector of means along
    # as mean of each row
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    # -- documentation at http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    G = -opt.matrix(np.eye(n)) # negative nxn id
    h = opt.matrix(0.0, (n, 1)) # n x 1 zero vector
    A = opt.matrix(1.0, (1, n)) # 1 x n 1-vector
    b = opt.matrix(1.0) # [1.0]

    ## Calculate efficient frontier weights using quadratic programming
    ## https://en.wikipedia.org/wiki/Quadratic_programming
    # Minimizes (1/2) w^T * S * w - q pbar^T * w for weights w
    # subject to the constrains:
    # G * w <= h (component-wise) -- positive weights
    # and A*w = b -- Sum of weights == 1
    # where S is the covariance matrix,
    # pbar are the mean returns,
    # q is a measure of risk tolerance
    portfolios = [solvers.qp(S, -q*pbar, G, h, A, b)['x'] for q in qs]

    ## Could replace above with Lagrange multiplier (or convex hull)

    # Calculate risks and mean returns for frontier
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Calculate quadratic for the frontier curve
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # Calculate the optimal portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks

def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading).
    Use this method to set up any bookkeeping variables.

    The context object is passed to all the other methods in your algorithm.

    Parameters

    context: An initialized and empty Python dictionary that has been
             augmented so that properties can be accessed using dot
             notation as well as the traditional bracket notation.

    Returns None
    '''
    # Register history container to keep a window of the last 100 prices.
    add_history(100, '1d', 'price')
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0

def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's
    securities.

    Parameters

    data: A dictionary keyed by security id containing the current
          state of the securities in the algo's universe.

    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state
             variables defined.

    Returns None
    '''
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = history(100, '1d', 'price').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Perform Markowitz-style portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T)
        # Rebalance portfolio accordingly
        for stock, weight in zip(prices.columns, weights):
            order_target_percent(stock, weight)
    except ValueError:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass

# Instantinate algorithm
algo = TradingAlgorithm(initialize=initialize,
                        handle_data=handle_data)

# Run algorithm
results = algo.run(data)
results.portfolio_value.plot()

# return_vec = np.random.randn(NUM_ASSETS, NUM_OBSERVATIONS)
# means, stds = np.column_stack([random_portfolio(return_vec) for _ in xrange(NUM_PORTFOLIOS)])
# weights, returns, risks = optimal_portfolio(return_vec)

# print weights
# print sum(weights)
# print weights*CAPITAL

# plt.plot(stds, means, 'o', markersize=5)
# plt.plot(risks, returns, 'y-o')
# plt.xlabel('std')
# plt.ylabel('mean')
# plt.title('Mean and standard deviation of returns of randomly generated portfolios')
# plt.show()