######################################
# Bermudan interest-rate swap option #
######################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from calibration import simulate_rate_numeraire_paths


#%%##############
# 1. Parameters #
#################

market_data = [
    ["1M", 0.0372], 
    ["3M", 0.0368],
    ["6M", 0.0379],
    ["1Y", 0.0386],
    ["2Y", 0.0413],
    ["3Y", 0.0418],
    ["5Y", 0.0427],
    ["7Y", 0.0441],
]


tenors = [row[0] for row in market_data]
tenors = [
    int(x[:-1]) / 12 if x.endswith("M")
    else int(x[:-1])
    for x in tenors
]

yield_data = np.array([row[1] for row in market_data], dtype=float)

market_bond_price = np.zeros(len(yield_data))
for i in range(len(yield_data)):
    market_bond_price[i] = np.exp(- yield_data[i] * tenors[i])


times = np.array([
    0.00,
    1.00,
    2.00,
    3.00,
    4.00,
    5.00
], dtype=float)


alpha = 0.03
sigma = 0.01

notional_amount = 10000
strike_rate = 0.04

num_total_samples = 10000

num_basis = 3
maturity_index = len(times)-2
final_exchange_date_index = len(times)-1


#%%##################
# 2. Sampling paths #
#####################

short_rate = np.zeros((num_total_samples, len(times)))
numeraire = np.zeros((num_total_samples, len(times)))
for i in range(num_total_samples):
    short_rate[i,:], numeraire[i,:] = simulate_rate_numeraire_paths(market_data, times, alpha, sigma)


yield_data = np.array([row[1] for row in market_data], dtype=float)
yield_curve = CubicSpline(tenors, yield_data, bc_type="natural")

yield_curve_times = yield_curve(times)
dy_t = yield_curve.derivative()(times)
instantaneous_forward = yield_curve_times + times * dy_t


#%%#############
# 3. Functions #
################

def bond_price(short_rate_at_valuation, valuation_time, maturity, time_0_bond_price_t, time_0_bond_price_T, instantaneous_forward_t, alpha, sigma):
    A = (1/alpha) * (1 - np.exp(- alpha * (maturity - valuation_time)))
    C = A * instantaneous_forward_t - (sigma**2/(4 * alpha)) * (1 - np.exp(- 2 * alpha * valuation_time)) * A * A
    return (time_0_bond_price_T/time_0_bond_price_t) * np.exp(-A * short_rate_at_valuation + C)

def instantaneous_forward_rate(t):
    y_t = yield_curve(t)
    dy_t = yield_curve.derivative()(t)
    return y_t + t * dy_t

basis_funcs = []
for i in range(num_basis):
    basis_funcs.append(lambda x, i=i: (x / strike_rate) ** i)

def swap_intrinsic_value(current_short_rate, valuation_time_index):
    swap_annuity = 0
    swap_rate = 0
    maturity = len(times)-1
    
    for i in range(valuation_time_index, maturity):
        swap_annuity += bond_price(short_rate_at_valuation = current_short_rate, 
                   valuation_time = times[valuation_time_index], 
                   maturity = times[i+1], 
                   time_0_bond_price_t = bond_prices[valuation_time_index], 
                   time_0_bond_price_T = bond_prices[i+1], 
                   instantaneous_forward_t = instantaneous_forward[valuation_time_index], 
                   alpha = alpha, 
                   sigma = sigma) \
            * (times[i+1] - times[i])
    
    bond = bond_price(short_rate_at_valuation = current_short_rate, 
               valuation_time = times[valuation_time_index], 
               maturity = times[maturity], 
               time_0_bond_price_t = bond_prices[valuation_time_index], 
               time_0_bond_price_T = bond_prices[maturity], 
               instantaneous_forward_t = instantaneous_forward[valuation_time_index], 
               alpha = alpha, 
               sigma = sigma)
    swap_rate = (1 - bond)/ swap_annuity
    
    return  np.maximum(notional_amount * swap_annuity * (swap_rate - strike_rate),0)


#%%##############################
# 4. Computing the strike price #
#################################

bond_prices = np.zeros(len(times))
for i in range(len(times)):
    bond_prices[i] = np.exp(- yield_curve_times[i] * times[i])

initial_swap_annuity = 0
for i in range(final_exchange_date_index):
    initial_swap_annuity += bond_price(short_rate_at_valuation = yield_curve(0), 
                               valuation_time = times[0], 
                               maturity = times[i+1], 
                               time_0_bond_price_t = bond_prices[0], 
                               time_0_bond_price_T = bond_prices[i+1], 
                               instantaneous_forward_t = instantaneous_forward[0], 
                               alpha = alpha, sigma = sigma) \
        * (times[i+1] - times[i])

bond_0 = bond_price(short_rate_at_valuation = yield_curve(0), 
           valuation_time = times[0], 
           maturity = times[final_exchange_date_index], 
           time_0_bond_price_t = bond_prices[0], 
           time_0_bond_price_T = bond_prices[final_exchange_date_index], 
           instantaneous_forward_t = instantaneous_forward[0], 
           alpha = alpha, 
           sigma = sigma)
strike_rate = (1 - bond_0)/ initial_swap_annuity


#%%######################
# 5. Backward induction #
#########################

intrinsic_value = np.zeros((num_total_samples, final_exchange_date_index))

for i in range(num_total_samples):
    for j in range(final_exchange_date_index):
        intrinsic_value[i][j] = swap_intrinsic_value(short_rate[i][j], j)

cashflow_matrix = np.zeros((num_total_samples, final_exchange_date_index))
cashflow_matrix[:, final_exchange_date_index-1] = intrinsic_value[:, final_exchange_date_index-1]

for j in reversed(range(1, final_exchange_date_index-1)):
    coeff_vec = np.zeros(num_basis)
    B_psi_V = np.zeros(num_basis)
    B_psi = np.zeros((num_basis, num_basis))
    X = np.array([])
    Y = np.array([])

    for i in range(num_total_samples):
        if intrinsic_value[i][j] != 0:
            future_cashflows = cashflow_matrix[i][j+1:]
            nonzero_index = np.nonzero(future_cashflows)[0]
            if len(nonzero_index) > 0:
                first_future_index = (nonzero_index[0]+(j+1))
                discounted_future_cashflow = (
                    (numeraire[i][j] * cashflow_matrix[i][first_future_index])/ numeraire[i][first_future_index]
                )
                X = np.append(X, short_rate[i][j])
                Y = np.append(Y, discounted_future_cashflow)
                
    length = len(X)

    for i in range(length):
        for q in range(num_basis):
            for r in range(num_basis):
                B_psi[q][r] += basis_funcs[q](X[i]) * basis_funcs[r](X[i])
            B_psi_V[q] += basis_funcs[q](X[i]) * Y[i]
    B_psi /= length
    B_psi_V /= length
    B_psi_inv = np.linalg.pinv(B_psi)
    coeff_vec = np.dot(B_psi_inv, B_psi_V)
    
    for i in range(num_total_samples):
        if intrinsic_value[i][j] != 0:
            vector = np.array([
                basis_funcs[m](short_rate[i][j])
                for m in range(num_basis)
            ])
            continuation_value = np.dot(coeff_vec, vector)
            if intrinsic_value[i][j] > continuation_value:
                cashflow_matrix[i][j] = (intrinsic_value[i][j])
                cashflow_matrix[i][j+1:final_exchange_date_index] = np.zeros(final_exchange_date_index -(j+1))
        else:
            cashflow_matrix[i][j] = 0


#%%################################
# 6. Computing the option premium #
###################################

option_premium_estimator = 0
for i in range(num_total_samples):
    for j in range(final_exchange_date_index):
        option_premium_estimator += cashflow_matrix[i][j] / numeraire[i][j]

option_premium_estimator = option_premium_estimator/num_total_samples

print(option_premium_estimator)




optimal_exercise_idx = np.full(num_total_samples, final_exchange_date_index)

for i in range(num_total_samples):
    nonzero_indices = np.nonzero(cashflow_matrix[i, :])[0]
    if len(nonzero_indices) > 0:
        optimal_exercise_idx[i] = nonzero_indices[0]

exposure_matrix = np.zeros((num_total_samples, final_exchange_date_index+1))

for i in range(num_total_samples):
    tau_idx = optimal_exercise_idx[i]
    
    for j in range(final_exchange_date_index+1):
        if j < tau_idx:
            val = swap_intrinsic_value(short_rate[i][j], j) if j < final_exchange_date_index else 0
            exposure_matrix[i][j] = max(val, 0)
        else:
            if tau_idx == final_exchange_date_index:
                exposure_matrix[i][j] = 0
            else:
                val = swap_intrinsic_value(short_rate[i][j], j) if j < final_exchange_date_index else 0
                exposure_matrix[i][j] = max(val, 0)


#%%###################
# 7. EE, PFE and CVA #
######################

EE = np.mean(exposure_matrix, axis=0)


def hazard_rate(t):
    return 0.02 

recovery_rate = 0.40  
LGD = 1.0 - recovery_rate  

survival_prob = np.zeros(final_exchange_date_index+1)
survival_prob[0] = 1.0

for j in range(1, final_exchange_date_index+1):
    t_prev = times[j-1]
    t_curr = times[j]
    dt = t_curr - t_prev
    avg_hazard = (hazard_rate(t_prev) + hazard_rate(t_curr)) / 2.0
    survival_prob[j] = survival_prob[j-1] * np.exp(-avg_hazard * dt)

discount_factors = np.zeros(final_exchange_date_index+1)
discount_factors[0] = 1.0
for j in range(1, final_exchange_date_index+1):
    discount_factors[j] = np.mean(1.0 / numeraire[:, j])


cva = 0.0
for j in range(1, final_exchange_date_index+1):
    delta_pd = survival_prob[j-1] - survival_prob[j]
    avg_discounted_EE = (
        discount_factors[j-1] * EE[j-1] + discount_factors[j] * EE[j]
    ) / 2.0
    cva += LGD * avg_discounted_EE * delta_pd

print(f"Calculated CVA: {cva:.4f}")


significance_level = np.array([
    99.0,
    97.5,
    95.0])

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(times, EE, label="Expected Exposure (EE)", color="blue", marker="o")
for i in range(len(significance_level)):
    plt.plot(times, np.percentile(exposure_matrix, significance_level[i], axis=0), label=f"{significance_level[i]}% PFE", linestyle="--", marker="x")
plt.title("Bermudan Swap Exposure Profiles")
plt.xlabel("Time (Years)")
plt.ylabel("Exposure")
plt.grid(True)
plt.legend()
plt.show()

