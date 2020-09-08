import pandas as pd
import numpy as np
import math
from scipy.stats import binom
import timeit
import pandas_market_calendars as mcal
from datetime import datetime
from dateutil import parser as datetime_parser
from dateutil.tz import tzutc,gettz

'''Imports SPY data and 3 Month TBill data'''
def spy_data_df():
    spy_data_raw = pd.read_csv('https://raw.githubusercontent.com/sambrazer/Binomial-Tree-Model/master/SPY.csv')
    tbill_data = pd.read_csv('https://raw.githubusercontent.com/sambrazer/Binomial-Tree-Model/master/Tbill.csv')
    tbill_data = tbill_data.rename(columns = {'Close': 'RF'})

    '''Want binomial trees to start on 3-1-2006 index at 3296. Chop spy_data_raw to just have that info.
    Add in RF, Up move, Down move, and Probability columns '''
    spy_data = spy_data_raw.iloc[3296:].reset_index().drop('index', axis = 1)
    spy_data = pd.merge(spy_data, tbill_data, how = 'inner', left_index = True, right_index = True).drop(['Date_y', 'Difference', 'LT-STD Running', 'Momentum ROC', 'Momentum ROC Squared',
           'Average ROC Momentum'], axis = 1).rename(columns = {'Date_x': 'Date'})
    spy_data['Date'] = pd.to_datetime(spy_data['Date'])
    spy_data['RF'] = spy_data['RF']/100
    spy_data['Up Move'] = np.exp(spy_data['23 Day Vol (20 Trading Days in a 4 Week Period)'])
    spy_data['Down Move'] = np.exp(-1*spy_data['23 Day Vol (20 Trading Days in a 4 Week Period)'])
    spy_data['P(U)'] = (np.exp(spy_data['RF']/360)-spy_data['Down Move'])/(spy_data['Up Move']- spy_data['Down Move'])
    spy_data['P(D)'] = 1 - spy_data['P(U)']
    return(spy_data)

    '''Initialize tree Panda and general Panda holding trees. Form iterative 23 step trees. Store in a list. Then calc probability of end nodes.
    Store nodes in a seperate df. Calc weighted value and store in list.'''

def initial_tree(spy_data):
    tree_list = []
    valuation_date = []
    start_date = []
    predicted_price = []
    actual_price = []
    for n in range(len(spy_data)-23):
        dates = np.array([date for date in spy_data['Date'][n:n+24]])
        current_tree_df = pd.DataFrame(columns = [dates], index = [i for i in range(24)])
        current_tree_df[current_tree_df.columns[0]][0] = spy_data['Close'][n]
        for i in range(24):
            if i == 0: continue
            up_move_price = current_tree_df[current_tree_df.columns[i-1]][0]*spy_data['Up Move'].iloc[n]
            current_tree_df[current_tree_df.columns[i]][0] = up_move_price
            for k in range(i):
                down_move_price = current_tree_df[current_tree_df.columns[i-1]][k]*spy_data['Down Move'].iloc[n]
                current_tree_df[current_tree_df.columns[i]][k+1] = down_move_price

        prob = []
        for i in range(len(current_tree_df[current_tree_df.columns[-1]])):
            prob.append(binom.pmf(23-i, len(current_tree_df[current_tree_df.columns[-1]])-1, spy_data['P(U)'].iloc[n]))
        prob_node_df = pd.DataFrame({'Predicted Price': current_tree_df[current_tree_df.columns[-1]], 'Probability': prob})
        prob_node_df['Weighted Value'] = prob_node_df['Predicted Price']*prob_node_df['Probability']
        actual_price_index = spy_data['Close'][spy_data['Date']==current_tree_df.columns[-1][0]].index[0]

        start_date.append(current_tree_df.columns[0][0])
        valuation_date.append(current_tree_df.columns[-1][0])
        predicted_price.append(prob_node_df['Weighted Value'].sum())
        actual_price.append(spy_data['Close'].iloc[actual_price_index])
        tree_list.append(current_tree_df)

        del prob
        del dates
        del current_tree_df
        del prob_node_df
        del actual_price_index


    '''Outputs the Predicted vs Actual Prices. Creates new columns with the nominal tracking error, % error based on modeled price.'''
    predicted_actual_df = pd.DataFrame({'Start Date': start_date, 'Valuation Date': valuation_date,'Predicted Price': predicted_price, 'Actual Price': actual_price})
    predicted_actual_df = predicted_actual_df.reset_index().drop('index', axis = 1)
    predicted_actual_df['Tracking Error'] = predicted_actual_df['Predicted Price'] - predicted_actual_df['Actual Price']
    predicted_actual_df['Error %'] = predicted_actual_df['Tracking Error']/predicted_actual_df['Predicted Price']
    return(predicted_actual_df)


'''Outputs the quantitative stats of model vs known prices'''
def stats_df(predictive_df):
    stats_df = predictive_df.describe()
    return(stats_df)

'''Outputs the predicted price of binomial tree given a valuation date. Need to fix this for speed, but I just reimport data and run Binomial
    model 1 time using the valuation date. Import market holidays to skip when running model. Can't just import the predictive_df because it
    is not valued to the valuation date. Only values up to valuation date - 23 days. Need to work with last day in the SPY data sheet as date
    to start valuation. Tasks: figure out how to import trading holidays, run model to value as of date given by user, should output a single
    int that is the predicted value of SPY given the valuation date.'''
def current_valuation(value_date):
    spy_data = spy_data_df()

    '''Use value date as start date for model. Use mcal library to pull dates. Given start date, pulls trading days for entire year then chops so that only
     24 trading days are utlized. Craft binomial tree from start date quant features.'''
    end_date_year = str(value_date.year + 1)
    end_date = pd.to_datetime(str(end_date_year + '/12/31'))
    nyse = mcal.get_calendar('NYSE')
    dates = [date.replace(tzinfo = None) for date in nyse.valid_days(start_date = value_date, end_date = end_date)[:24]]

    valuation_tree_df = pd.DataFrame(columns = [dates], index = [i for i in range(24)])
    valuation_tree_df[valuation_tree_df.columns[0]][0] = spy_data['Close'].loc[spy_data['Date'] == dates[0]].values[0]

    for i in range(24):
        if i == 0: continue
        up_move_price = valuation_tree_df[valuation_tree_df.columns[i-1]][0]*spy_data['Up Move'].loc[spy_data['Date'] == dates[0]].values[0]
        valuation_tree_df[valuation_tree_df.columns[i]][0] = up_move_price
        for k in range(i):
            down_move_price = valuation_tree_df[valuation_tree_df.columns[i-1]][k]*spy_data['Down Move'].loc[spy_data['Date'] == dates[0]].values[0]
            valuation_tree_df[valuation_tree_df.columns[i]][k+1] = down_move_price

    prob = []
    for i in range(len(valuation_tree_df[valuation_tree_df.columns[-1]])):
        prob.append(binom.pmf(23-i, len(valuation_tree_df[valuation_tree_df.columns[-1]])-1, spy_data['P(U)'].loc[spy_data['Date'] == dates[0]].values[0]))
    prob_node_df = pd.DataFrame({'Predicted Price': valuation_tree_df[valuation_tree_df.columns[-1]], 'Probability': prob})
    prob_node_df['Weighted Value'] = prob_node_df['Predicted Price']*prob_node_df['Probability']

    return(prob_node_df['Weighted Value'].sum())

def CI_95(stats_df):
    '''Create the 95% confidence interval Data frame based on the stats_df. Calced as mean error +/- 2 times STD of error.'''
    CI_columns= ['Mean Error', 'STD of Error', 'Upper Bound', 'Lower Bound']
    CI_95 = pd.DataFrame(columns = CI_columns)
    CI_95.loc[0] = [stats_df['Error %'].loc['mean'], stats_df['Error %'].loc['std'], stats_df['Error %'].loc['mean']+2*stats_df['Error %'].loc['std'], stats_df['Error %'].loc['mean']-2*stats_df['Error %'].loc['std']]
    return(CI_95)
