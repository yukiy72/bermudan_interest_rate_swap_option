import numpy as np
from scipy.interpolate import CubicSpline
# from scipy.interpolate import PchipInterpolator

def simulate_rate_numeraire_paths(market_data, exercisable_date, alpha, sigma):
    
    tenors = [row[0] for row in market_data]
    tenors = [
        int(x[:-1]) / 12 if x.endswith("M")
        else int(x[:-1])
        for x in tenors
    ]
    
    yield_data = np.array([row[1] for row in market_data], dtype=float)
    yield_curve = CubicSpline(tenors, yield_data, bc_type="natural")
    
    normal = np.random.normal(loc=0, scale=1, size=(len(exercisable_date)-1,2))
    short_rate = np.zeros(len(exercisable_date))
    short_rate[0] = yield_curve(0)
    
    y_t = yield_curve(exercisable_date)
    dy_t = yield_curve.derivative()(exercisable_date)
    instantaneous = y_t + exercisable_date * dy_t
    
    mu = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        mu[i] = instantaneous[i+1] - np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])) * instantaneous[i] \
            + (sigma**2 / alpha**2) \
                * (1 + np.exp(- 2 * alpha * exercisable_date[i+1]) - np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])) - np.exp(- alpha * (exercisable_date[i+1] + exercisable_date[i])))
    
    variance_r = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        variance_r[i] = (sigma**2/(2*alpha)) * (1 - np.exp(-2* alpha * (exercisable_date[i+1] - exercisable_date[i])))

    for i in range(len(exercisable_date)-1):
        short_rate[i+1] = np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])) * short_rate[i] + mu[i] + np.sqrt(variance_r[i]) * normal[i,0]

    bond_price_on_exercisable_dates = np.zeros(len(exercisable_date))
    for i in range(len(exercisable_date)):
        bond_price_on_exercisable_dates[i] = np.exp(- yield_curve(exercisable_date[i]) * exercisable_date[i])

    mean_Y = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        A = (1/alpha) * (1 - np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])))
        mean_Y[i] = A * (short_rate[i] - instantaneous[i]) + np.log(bond_price_on_exercisable_dates[i]/bond_price_on_exercisable_dates[i+1]) \
            + (sigma**2/(2*alpha**2)) * (exercisable_date[i+1] - exercisable_date[i] - A - (alpha/2) * np.exp(-2 * alpha * exercisable_date[i]) * A * A)

    variance_Y = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        A = (1/alpha) * (1 - np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])))
        variance_Y[i] = (sigma**2/alpha**2) * ((exercisable_date[i+1] - exercisable_date[i]) - A - (alpha/2) * A * A)

    covariance_rY = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        A = (1/alpha) * (1 - np.exp(- alpha * (exercisable_date[i+1] - exercisable_date[i])))
        covariance_rY[i] = (sigma**2/2) * A * A

    correlation_rY = np.zeros(len(exercisable_date)-1)
    for i in range(len(exercisable_date)-1):
        correlation_rY[i] = covariance_rY[i] / (np.sqrt(variance_r[i]) * np.sqrt(variance_Y[i]))

    Y = np.zeros(len(exercisable_date))
    # initial_numeraire = 0
    
    for i in range(len(exercisable_date)-1):
        Y[i+1] = Y[i] + mean_Y[i] + np.sqrt(variance_Y[i]) * (correlation_rY[i] * normal[i,0] + np.sqrt(1 - correlation_rY[i]**2) * normal[i,1])

    numeraire = np.exp(Y)

    return short_rate, numeraire

"""
market_data = [
    ["1M", 0.0372], 
    ["3M", 0.0368],
    ["6M", 0.0379],
    ["1Y", 0.0386],
    ["2Y", 0.0413],
    ["3Y", 0.0418],
    ["5Y", 0.0427],
    ["7Y", 0.0441],
    ["10Y", 0.0456],
    ["20Y", 0.0506], 
    ["30Y", 0.0507]
]


exercisable_date = np.array([
    1.00,
    2.00,
    3.00,
    4.00,
    5.00
], dtype=float)


alpha = 0.2
sigma = 0.01

import matplotlib.pyplot as plt
short_rate, numeraire = simulate_rate_numeraire_paths(market_data, exercisable_date, alpha, sigma)
plt.plot(exercisable_date, short_rate)
plt.plot(exercisable_date, 1/numeraire)

"""






