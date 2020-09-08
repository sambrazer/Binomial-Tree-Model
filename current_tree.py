import pandas as pd
import numpy as np
import math
from scipy.stats import binom
import binomial_tree as tree
import timeit
import pandas_market_calendars as mcal
from datetime import datetime
from dateutil import parser as datetime_parser
from dateutil.tz import tzutc,gettz

'''Utilizing the quatitative information outputted from CI_95 and the predicted price given the current valuation date, constructs the iron condor legs
WITHOUT drift.'''
def no_drift_iron_condor(CI_95, predicted_price):
    short_put_leg = round(((1 + CI_95['Lower Bound'][0])*predicted_price)/5)*5
    long_put_leg = short_put_leg - leg_spread
    short_call_leg = round(((1 + CI_95['Upper Bound'][0])*predicted_price)/5)*5
    long_call_leg = short_call_leg + leg_spread
    no_drift_iron_condor = pd.DataFrame(columns = [iron_condor_column_labels], index = [iron_condor_index])
    no_drift_iron_condor.iloc[0] = [long_put_leg, short_put_leg, short_call_leg, long_call_leg]
    return(no_drift_iron_condor)

'''Utilizing the quatitative information outputted from CI_95 and the predicted price given the current valuation date, constructs the iron condor legs
WITH drift.'''
def drift_iron_condor(CI_95, predicted_price, spy_data, momentum_reduction):
    short_put_leg = round(((1 + CI_95['Lower Bound'][0])*predicted_price*(1 + (spy_data['23 Day Momentum'].loc[spy_data['Date'] == value_date].values[0]/momentum_reduction)))/5)*5
    long_put_leg = short_put_leg - leg_spread
    short_call_leg = round(((1 + CI_95['Upper Bound'][0])*predicted_price*(1 + (spy_data['23 Day Momentum'].loc[spy_data['Date'] == value_date].values[0]/momentum_reduction)))/5)*5
    long_call_leg = short_call_leg + leg_spread
    drift_iron_condor = pd.DataFrame(columns = [iron_condor_column_labels], index = [iron_condor_index])
    drift_iron_condor.iloc[0] = [long_put_leg, short_put_leg, short_call_leg, long_call_leg]
    return(drift_iron_condor)


'''Ask user for date that next model starts (date given relates to last known price model assumes). Runs model to obtain stats, then runs model
on current valuation date. Outputs modeled price. Asks user how large the spread is between long and short legs. If not a multiple of 5, rounds to
the nearest 5. Also asks users how much they want to reduce the momentum'''
spy_data = tree.spy_data_df()
predictive_df = tree.initial_tree(spy_data)
stats_df = tree.stats_df(predictive_df)
CI_95 = tree.CI_95(stats_df)

iron_condor_column_labels = ['Long Put', 'Short Put', 'Short Call', 'Long Call']
iron_condor_index = ['Strike', 'Price']
value_date_str = input('Enter date for valuation (yyyy/mm/dd formate): ')
leg_spread_input = float(input('Enter the desired spread between long and short legs (multiple of 5): '))
leg_spread = round(leg_spread_input/5)*5
value_date = pd.to_datetime(value_date_str)
print('Current momentum:', spy_data['23 Day Momentum'].loc[spy_data['Date'] == value_date].values[0])
momentum_reduction = float(input('Enter the desired momentum reduction: '))

'''Grabs needed quant information from bionmial_tree.py and uses it to generate the iron condor with drift and the iron condor without drift legs.'''
predicted_price = tree.current_valuation(value_date)
no_drift_model = no_drift_iron_condor(CI_95, predicted_price)
drift_model = drift_iron_condor(CI_95, predicted_price, spy_data, momentum_reduction)

print(no_drift_model)
print(drift_model)
