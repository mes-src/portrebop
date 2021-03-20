import pandas as pd, numpy as np
import pypfopt 


import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))



opt = pypfopt.EfficientFrontier(expected_returns, 
                                cov_matrix, 
                                solver='CVXOPT').efficient_return(mu_x) 


opt = pypfopt.EfficientFrontier(expected_returns,
                                cov_matrix,
                                solver='CVXOPT').efficient_risk(sigma_x) 

opt = pypfopt.EfficientFrontier(expected_returns, 
                                cov_matrix,
                                solver='CVXOPT').max_sharpe(0) 
print(opt)

