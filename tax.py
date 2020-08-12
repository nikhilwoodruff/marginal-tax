import numpy as np
import pandas as pd
from timeit import timeit

test_data = np.random.rand(1000000) * 300000

def test0():
    brackets = np.array([0, 12500, 50000, 150000])
    rates = np.array([0, 0.2, 0.4, 0.45])

    def income_tax(income, brackets, rates):
        df_tax = pd.DataFrame({'brackets': brackets, 'rates': rates})
        df_tax['base_tax'] = df_tax.brackets.\
            sub(df_tax.brackets.shift(fill_value=0)).\
            mul(df_tax.rates.shift(fill_value=0)).cumsum()
        rows = df_tax.brackets.searchsorted(income, side='right') - 1
        income_bracket_df = df_tax.loc[rows].reset_index(drop=True)
        return pd.Series(income).sub(income_bracket_df.brackets).\
            mul(income_bracket_df.rates).add(income_bracket_df.base_tax)
    
    income_tax(test_data, brackets, rates)

def test1():
    bands = np.array([0, 12500, 50000, 150000])
    rates = np.array([0, 0.2, 0.4, 0.45])

    def tax(incomes, bands, rates):
        bands = np.append(bands, np.inf)
        incomes_ = np.broadcast_to(incomes, (bands.shape[0] - 1, incomes.shape[0]))
        amounts_in_bands = np.clip(incomes_.transpose(), bands[:-1], bands[1:]) - bands[:-1]
        taxes = rates * amounts_in_bands
        total_taxes = taxes.sum(axis=1)
        return total_taxes
    
    tax(test_data, bands, rates)

time0 = timeit(test0, number=10)
time1 = timeit(test1, number=10)
speed_percent_change = (time1 - time0) / time0
print(f'NumPy method changes time taken by {speed_percent_change * 100:.2f}%.')