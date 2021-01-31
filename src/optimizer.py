import pandas as pd
import json
from pypfopt import EfficientFrontier

# https://github.com/robertmartin8/PyPortfolioOpt

with open('../resources/stock_data.json') as file:
    data = json.load(file)


def get_cov(portfolio_data, symbol_a, symbol_b):
    assert portfolio_data[symbol_a]["correlations"][symbol_b] == portfolio_data[symbol_b]["correlations"][symbol_a]
    volatility_a = portfolio_data[symbol_a]["volatility"]
    volatility_b = portfolio_data[symbol_b]["volatility"]
    correlation_a_b = portfolio_data[symbol_b]["correlations"][symbol_a]
    covariance_a_b = volatility_a * volatility_b * correlation_a_b
    return covariance_a_b


def get_cov_matrix(portfolio_data):
    result = []
    for first_key in portfolio_data.keys():
        cov_row = []

        for second_key in portfolio_data.keys():
            if first_key == second_key:
                cov_row.append(portfolio_data[first_key]["volatility"])
            else:
                cov_row.append(get_cov(data, first_key, second_key))
        result.append(cov_row)
    return result


def get_returns(portfolio_data):
    result = []
    for key in data.keys():
        result.append(portfolio_data[key]["return"])
    return result


def change_key_in_ordered_dict(ordered_dict, old_key, new_key):
    for _ in range(len(ordered_dict)):
        k, v = ordered_dict.popitem(False)
        ordered_dict[new_key if old_key == k else k] = v


def fix_keys(ordered_dict, dict_with_keys):
    counter = 0
    for key in dict_with_keys.keys():
        change_key_in_ordered_dict(ordered_dict, counter, key)
        counter += 1


def optimise(portfolio_data):
    returns = pd.Series(get_returns(portfolio_data))
    covariance_matrix = pd.DataFrame(get_cov_matrix(portfolio_data))

    ef = EfficientFrontier(returns, covariance_matrix)
    ef.min_volatility()  # max_sharpe or min_volatility
    cleaned_weights = ef.clean_weights()
    fix_keys(cleaned_weights, portfolio_data)
    print(cleaned_weights)
    ef.save_weights_to_file("weights.csv")  # saves to file
    ef.portfolio_performance(verbose=True)


if __name__ == '__main__':
    optimise(data)
