#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:46:57 2022

@author: chenqi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.optimize import leastsq 
import scipy as sp   
import math
import os
import re
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

# Load data, China: 1.22 -- 07.02     Italy: 1.31 -- 7.02

def readDeathData(country):
    death_data = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    data = death_data[(death_data['Country/Region'] == country)].iloc[:,4:]
    # & (death_data['Province/State'].isna())
    return data.sum()

ChinaDeathData = readDeathData('China')
ItalyDeathData = readDeathData('Italy')[9:]

def readConfirmedData(country):
    recover_data = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    data = recover_data[(recover_data['Country/Region'] == country)].iloc[:,4:]
    return data.sum()

ChinaConfirmedData = readConfirmedData('China')
ItalyConfirmedData = readConfirmedData('Italy')[9:]

def readRecoveredData(country):
    recover_data = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    data = recover_data[(recover_data['Country/Region'] == country)].iloc[:,4:]
    return data.sum()

ChinaRecoveredData = readRecoveredData('China')
ItalyRecoveredData = readRecoveredData('Italy')[9:]


# Get more data for China     2020.1.22 --- 2022.7.9      900 in total

def readDeathDataNew(country):
    death_data = pd.read_csv('data_new/time_series_covid19_deaths_global.csv')
    data = death_data[(death_data['Country/Region'] == country)].iloc[:,4:]
    # & (death_data['Province/State'].isna())
    return data.sum()

ChinaDeathDataNew = readDeathDataNew('China')

def readConfirmedDataNew(country):
    recover_data = pd.read_csv('data_new/time_series_covid19_confirmed_global.csv')
    data = recover_data[(recover_data['Country/Region'] == country)].iloc[:,4:]
    return data.sum()

# Confiremed data of China is different than the original data since 2020.3.23

ChinaConfirmedDataNew = readConfirmedDataNew('China')
ItalyConfirmedDataNew = readConfirmedDataNew('Italy')[9:]

def readRecoveredDataNew(country):
    recover_data = pd.read_csv('data_new/time_series_covid19_recovered_global.csv')
    data = recover_data[(recover_data['Country/Region'] == country)].iloc[:,4:]
    return data.sum()
    recover_data = pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    data = recover_data[(recover_data['Country/Region'] == country)].iloc[:,4:]
    return data.sum()

ChinaRecoveredDataNew = readRecoveredDataNew('China')


# Create population(N) dictionnary for a few countries

popu_dict = {'Brazil':210867954,'Korea, South':51269185,'Germany':82927922,'United Kingdom':66573504,'US':326766748,'China':1400050000, 'Italy':60482200, 'France':65233271, 'Spain':46397452 ,'Canada':36953765,'sars_bj': 14223000}

# Create function to get the actural number of I, R, D, C

def load_IR(name, N, t,T=None):
    if name == 'sars_bj':
        sars_bj = pd.read_excel('sars_bj.xlsx')
        R = sars_bj['R']
        D = sars_bj['D']
        C = sars_bj['C']
        R = R.tolist()
        D = D.tolist()
        C = C.tolist()
    else:
        R = readRecoveredData(name)
        D = readDeathData(name)
        C = readConfirmedData(name) 
        R = R.tolist()[t:T]
        D = D.tolist()[t:T]
        C = C.tolist()[t:T]
    R = np.array(R)
    D = np.array(D)
    C = np.array(C)
    I = C - R - D
    print("load IR:", I, R)
    return I,R,D,C

ChinaInfectedData = load_IR('China', popu_dict['China'], 0)[0]
ItalyInfectedData = load_IR('Italy', popu_dict['Italy'], 9)[0]

# Get extra data for China

def load_IR_New(name, N, t,T=None):
    if name == 'sars_bj':
        sars_bj = pd.read_excel('sars_bj.xlsx')
        R = sars_bj['R']
        D = sars_bj['D']
        C = sars_bj['C']
        R = R.tolist()
        D = D.tolist()
        C = C.tolist()
    else:
        R = readRecoveredDataNew(name)
        D = readDeathDataNew(name)
        C = readConfirmedDataNew(name) 
        R = R.tolist()[t:T]
        D = D.tolist()[t:T]
        C = C.tolist()[t:T]
    R = np.array(R)
    D = np.array(D)
    C = np.array(C)
    I = C - R - D
    print("load IR:", I, R)
    return I,R,D,C

ChinaInfectedDataNew = load_IR_New('China', popu_dict['China'], 0)[0]
ItalyInfectedDataNew = load_IR_New('Italy', popu_dict['Italy'], 9)[0]


def SSE(I_t, I):
    sse = (I_t[:len(I)] - I)**2
    return sse.sum()

# Model Solution Runge-Kutta methods
def RungeKutta(beta, gamma, N, S_0, I_0, R_0, t, days = 24):

    X = np.arange(t, days)
    I_t = np.zeros(days)
    S_t = np.zeros(days)
    R_t = np.zeros(days)
    I_t[0] = I_0
    S_t[0] = S_0
    R_t[0] = R_0
    h = 1  
    for i in range(1, days): 
        k11 = -beta * S_t[i-1] * I_t[i-1] / N
        k21 = beta * S_t[i-1] * I_t[i-1] / N - gamma * I_t[i-1]
        k31 = gamma * I_t[i-1]

        k12 = -beta * (S_t[i-1] + h / 2 * k11) * (I_t[i-1] + h / 2 * k21) / N
        k22 = beta * (S_t[i-1] + h / 2 * k11) * (I_t[i-1] + h / 2 * k21) / N - gamma * (I_t[i-1] + h / 2 * k21)
        k32 = gamma * (I_t[i-1] + h / 2 * k21)

        k13 = -beta * (S_t[i-1] + h / 2 * k12) * (I_t[i-1] + h / 2 * k22) / N
        k23 = beta * (S_t[i-1] + h / 2 * k12) * (I_t[i-1] + h / 2 * k22) / N - gamma * (I_t[i-1] + h / 2 * k22)
        k33 = gamma * (I_t[i-1] + h / 2 * k22)

        k14 = -beta * (S_t[i-1] + h * k13) * (I_t[i-1] + h * k23) / N
        k24 = beta * (S_t[i-1] + h * k13) * (I_t[i-1] + h * k23) / N - gamma * (I_t[i-1] + h * k23)
        k34 = gamma * (I_t[i-1] + h * k23)

        S_t[i] = S_t[i-1] + h / 6 * (k11 + 2 * k12 + 2 * k13 + k14)
        I_t[i] = I_t[i-1] + h / 6 * (k21 + 2 * k22 + 2 * k23 + k24)
        R_t[i] = R_t[i-1] + h / 6 * (k31 + 2 * k32 + 2 * k33 + k34)
    return S_t[t:],I_t[t:],R_t[t:]

def findOptimalParameterWithRatio(I, R, N, beta_ini,gamma_ini):
    S_0 = N - I[0] - R[0]
    I_0 = I[0]
    R_0 = R[0]
    t = 0
    step = 0.01
    minSSe = float("inf")
    result = []
    beta_0,gamma_0 = beta_ini, gamma_ini
    minSSe = float("inf")
    beta = beta_ini - 1 
    gamma = gamma_ini - 1
    if beta == 0: beta = 0.01
    if gamma == 0: gamma = 0.01
    result_list = []
    I_opt = I
    while beta < beta_0:
        gamma = gamma_0 - 1
        if gamma == 0: gamma = 0.01
        while gamma < beta:
            if beta == 0 or gamma == 0 or beta == gamma: continue
            S_t,I_t,R_t = RungeKutta(beta, gamma, N, S_0, I_0, R_0, t)
            sse = SSE(I_t,I)
            #period = I_t.argmax() - dailyIncrease_p.argmax()
            result_list.append((beta,gamma,sse))
            if sse < minSSe:
                minSSe = sse
                result = (beta, gamma)
                I_opt = I_t[:len(I)]
            gamma = gamma + step
        beta = beta + step
    print('r0 = ' + str(result[0]/result[1]) +' beta = ' + str(result[0]) + ' gamma = ' + str(result[1]) + ' sse =' + str(minSSe))
    return result[0], result[1], minSSe, result_list

def findOptimalParameterWithRatio2(I, R, N, beta_ini,gamma_ini):
    S_0 = N - I[0] - R[0]
    I_0 = I[0]
    R_0 = R[0]
    t = 0
    step = 0.01
    minSSe = float("inf")
    result = []
    beta_0,gamma_0 = beta_ini, gamma_ini
    minSSe = float("inf")
    beta = beta_ini - 1 
    gamma = gamma_ini - 1
    if beta == 0: beta = 0.01
    if gamma == 0: gamma = 0.01
    result_list = []
    I_opt = I
    while gamma < gamma_0:
        beta = beta_0 - 1
        if beta == 0: beta = 0.01
        while beta < gamma:
            if beta == 0 or gamma == 0 or beta == gamma: continue
            S_t,I_t,R_t = RungeKutta(beta, gamma, N, S_0, I_0, R_0, t)
            sse = SSE(I_t,I)
            #period = I_t.argmax() - dailyIncrease_p.argmax()
            result_list.append((beta,gamma,sse))
            if sse < minSSe:
                minSSe = sse
                result = (beta, gamma)
                I_opt = I_t[:len(I)]
            beta = beta + step
        gamma = gamma + step
    print('r0 = ' + str(result[0]/result[1]) +' beta = ' + str(result[0]) + ' gamma = ' + str(result[1]) + ' sse =' + str(minSSe))
    return result[0], result[1], minSSe, result_list

def search(I,R,N,range1,range2):
    minSSE = float('inf')
    result = []
    result_all = []
    result_opt = []
    for i in range(range1,range2):
        print('initial value : ',i)
        beta,gamma,sse,result_list = findOptimalParameterWithRatio(I,R,N,i,i)

        beta2,gamma2,sse2,result_list2 = findOptimalParameterWithRatio2(I,R,N,i,i)
        if sse2 < sse:
            beta = beta2
            gamma = gamma2
            sse = sse2
            result_list = result_list2
   
        result_all = result_all + result_list
        result_opt.append((beta,gamma,sse))
        if sse < minSSE:
            result = (beta,gamma)
            minSSE = sse
    print("best param: ", result, "minSSE: ",minSSE)
    S_t,I_t,R_t = RungeKutta(result[0], result[1], N, N - I[0] - R[0], I[0], R[0], 0)
    return result_all,result_opt,I_t
    
def create_assist_date(datestart = None,dateend = None):
    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
    dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y-%m-%d'))
    while datestart<dateend:
        datestart+=datetime.timedelta(days=+1)
        date_list.append(datestart.strftime('%Y-%m-%d'))
    return date_list

def predictPlot(I, I_t):
    date_X = create_assist_date("2020-1-22", "2021-10-01")
    X = np.arange(0, len(I_t))
    ax  = plt.figure(figsize=(13, 8))
    sns.lineplot(X[:len(I_t)],I_t,label="Predict Infected")
    sns.lineplot(X[:len(I)], I, label = 'Current Infected')
    plt.xlabel('Date')
    plt.ylabel('Number of active infections')
    plt.title('SIR Model')

def save_result(i,j, result_all, result_opt, I_t, country_name,size):
    dirs = '\\data\\'
    dir_path = os.path.dirname(os.path.abspath('__file__')) + dirs
    dir_path = dir_path+country_name+'\\'+str(size)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df_all = pd.DataFrame(result_all,columns=['beta', 'gamma', 'sse'])
    df_all.to_csv(dir_path + '/'+str(i) + '_'+str(j)+'_all.csv')
    df_opt = pd.DataFrame(result_opt,columns=['beta', 'gamma', 'sse'])
    df_opt.to_csv(dir_path + '/'+str(i) + '_'+str(j)+'_opt.csv')
    df_I = pd.DataFrame(I_t,columns=['I_t'])
    df_I.to_csv(dir_path+ '/'+str(i) + '_'+str(j)+'_It.csv')

# This function was defined incorrectly because the argument size is required in save_result

# def predictI(country,t):
#     I,R,D,C = load_IR(country, popu_dict[country], t)
#     i = 0 
#     j = 6
#     I_t_list = []
#     I_t_all = []
#     for i in range(0, len(I)- 6):
#         if i == len(I) - 7:
#             j = len(I) - 1
#         else:
#             j = i + 6 
#         result_all,result_opt,I_t = search(I[i:j],R[i:j],popu_dict[country],1,2)
#         save_result(i,j, result_all, result_opt, I_t, country)
#         I_t_list.append(I_t)
#         if i == len(I) - 7:
#             I_t_all = I_t_all + list(I_t) 
#         else:
#             I_t_all = I_t_all + list(I_t[:7])
#     return I_t_list, I_t_all

def oneDayError(I,I_p):
    error = ((I_p[:len(I)] - I)/I)
    X = np.arange(0, len(error))
    ax  = plt.figure(figsize=(13, 8))
    sns.lineplot(X[:len(error)],error,label="Error")

# Correct the window size 

def read_I_t(I,country, size):
    I_t_list = []
    I_t_all = []
    I_t_p = list(I[0:size])
    I_t_p
    for i in range(0, len(I)- size + 1):
        if i == len(I) - size:
            j = len(I) - 1
        else:
            j = i + size - 1 
        file_name = str(i) + '_'+str(j)+'_It.csv'
        a = pd.read_csv('data/'+country+'/'+file_name,engine='python')
        I_t = list(a['I_t'])
        I_t_list.append(I_t)

        if i == len(I) - size:
            I_t_all = I_t_all + I_t
            I_t_p = I_t_p + I_t[size:]
        else:
            I_t_all = I_t_all + I_t[:1]
            I_t_p.append(I_t[size])
    return I_t_p,I_t_all

# Correct the window size 

def read_opt(I,country,size):
    beta_list = []
    gamma_list = []
    for i in range(0, len(I)- size + 1):
        if i == len(I) - size:
            j = len(I) - 1
        else:
            j = i + size - 1 
        file_name = str(i) + '_'+str(j)+'_opt.csv'
        a = pd.read_csv('data/'+country+'/'+file_name,engine='python')
        #opt = a.iloc[a['sse'].idxmin(),][1:3].tolist()
        opt = a.iloc[0,1:3].tolist()
        beta_list.append(opt[0])
        gamma_list.append(opt[1])
    beta_list = np.array(beta_list,dtype="float64")
    gamma_list = np.array(gamma_list,dtype="float64")
    return beta_list,gamma_list

def plot(data,label=None):
    X = np.arange(0, len(data))
    ax  = plt.figure(figsize=(13, 8))
    sns.lineplot(X[:len(data)],data,label=label)

# Parameter evaluation
# Add T to control the length of actual number of I

def predictI_bySize(country, t, size, T):
    I,R,D,C = load_IR(country, popu_dict[country], t, T)
    I_t_p = list(I[0:size])
    i = 0 
    j = size - 1
    for i in range(0, len(I)- size + 1):
        if i == len(I) - size:
            j = len(I) - 1
        else:
            j = i + size - 1 
        result_all,result_opt,I_t = search(I[i:j],R[i:j],popu_dict[country],1,2)
        save_result(i,j, result_all, result_opt, I_t, country,size)
        if i == len(I) - size:
            I_t_p = I_t_p + list(I_t[size:])
        else:
            I_t_p.append(I_t[size])
    predictPlot(I,I_t_p)

ChinaPredictI = predictI_bySize('China', 0, 7)
ChinaPredictI_3 = predictI_bySize('China', 0, 3)
ChinaPredictI_4 = predictI_bySize('China', 0, 4)

# For Italy, 3 is the optimal size
ItalyPredictI_3 = predictI_bySize('Italy', 9, 3)

# Set days = 24 in RungeKutta so that we can predict the following 24 I.
ChinaPredictI_new_4 = predictI_bySize(country = 'China', t = 0, size = 4, T = 140)


# Find the top 5 smallest values of sse
# n stands for the number of smallest sse you want to get. Size means the window size.
# days stand for the number of days you want to see the trend, len(days) = len(param).

def processData(n, size):
    result = pd.DataFrame()
    for i in range(0, 162 - size + 2):
        processedData = pd.read_csv(f'data_Size_{size}/China/{i}_{i+size-1}_all.csv')
        processedData = processedData.nsmallest(n, 'sse', 'all')
        result = result.append(processedData, ignore_index=True)
    return result

size_4 = processData(5, 4)
size_7 = processData(5, 7)

def plotTrend(days, param, n):
    for i in range(0,n):
        x = np.arange(0, days)
        labels = ['smallest', '2nd', '3rd', '4th', '5th']
        #plt.plot(x, param[i:len(param):n], label = labels[i])
        plt.scatter(x, param[i:len(param):n], s = 1, label = labels[i])
    plt.xlabel('days')
    plt.ylabel('param')
    plt.title(f'the trend of {n} smallest sse')
    plt.legend(loc = 0, prop = {'size':4})
    plt.savefig('param trend', dpi = 600)

# Plot the trend for size 4 
plotTrend(160, size_4['beta'], 5)
plotTrend(160, size_4['gamma'], 5)
    
# Plot the trend for size 7
plotTrend(157, size_7['beta'], 5)
plotTrend(157, size_7['gamma'], 5)


# Parameter prediction
from sklearn import  linear_model
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.metrics import mean_squared_error

minMSE = float('inf')
opt = 0

def predictParam(size, param):
    datasets_X = np.arange(0, len(param)).reshape([len(param),1])
    param_p = list(param[0:size])
    for j in range(size, len(param)):
        i = j - size + 1
        nextY = paramPredict(datasets_X[i:j], param[i:j])
        param_p.append(nextY)
    param_p = np.array(param_p)
    print('mse:',mean_squared_error(param_p, param),SSE(param,param_p))
    
    X = np.arange(0, len(param))
    ax  = plt.figure(figsize=(13, 8))
    sns.lineplot(X,param,label="param(t)")
    sns.lineplot(X, param_p, label = 'predict')
    plt.xlabel('Date')
    plt.ylabel('number')
    return param_p



def paramPredict(datasets_X,datasets_Y):
    poly_reg =PolynomialFeatures(degree=2)
    X_ploy =poly_reg.fit_transform(datasets_X)
    lin_reg_2=linear_model.LinearRegression()
    lin_reg_2.fit(X_ploy,datasets_Y)
    y_predict = lin_reg_2.predict(poly_reg.fit_transform(datasets_X))
    nextX = np.array(datasets_X[-1][0]+1).reshape([1,1])
    nextY = lin_reg_2.predict(poly_reg.fit_transform(nextX))
    return nextY[0]

def polyPredict(datasets_X,datasets_Y):
    poly_reg =PolynomialFeatures(degree=2)
    X_ploy =poly_reg.fit_transform(datasets_X)
    lin_reg_2=linear_model.LinearRegression()
    lin_reg_2.fit(X_ploy,datasets_Y)
    y_predict = lin_reg_2.predict(poly_reg.fit_transform(datasets_X))
    mse = mean_squared_error(datasets_Y, y_predict)
    print('mse: ', mse,SSE(datasets_Y, y_predict))
    plt.scatter(datasets_X,datasets_Y,color='red')
    plt.plot(datasets_X,y_predict,color='blue')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.show()
    return mse


# Get r0 and dif

def parameterPlot(beta_list, gamma_list):
    r0 = []
    dif = []
    for i in range(0,len(beta_list)):
        r0.append(beta_list[i]/gamma_list[i])
        dif.append(beta_list[i] - gamma_list[i])
    return r0, dif

# Correct some codes in this chunk
        
def saveParam(country, size,t, T):
    # Note, different country should be different t in load_IR function
    I,R,D,C = load_IR(country,popu_dict[country], t, T)
    beta_list,gamma_list = read_opt(I,country,size)
    r0,dif = parameterPlot(beta_list,gamma_list)
    where_are_NaNs = np.isnan(r0)
    if any(where_are_NaNs): # add if condition
        r0[np.argwhere(where_are_NaNs)] = 0
    where_are_NaNs = np.isnan(dif)
    if any(where_are_NaNs): # add if condition
        dif[np.argwhere(where_are_NaNs)] = 0       
    beta_list = beta_list[t:]
    gamma_list = gamma_list[t:]
    r0 = r0[t:]
    dif = dif[t:]
    I = I[t:]
    plot(r0)
    plot(dif)
    r0_p = predictParam(size, r0)
    dif_p = predictParam(size, dif)
    print("s", len(r0), len(r0_p), len(I))
    dirs = '\\data\\'
    dir_path = os.path.dirname(os.path.abspath('__file__')) + dirs
    dir_path = dir_path+country+'\\'
    date_X = create_assist_date("2020-1-22", "2021-10-01")
    date_X = date_X[t:]
    # Set the lower bound = 3 for China, 2 for Italy
    data = np.vstack((date_X[3:len(I)],beta_list,gamma_list,r0,dif,r0_p,dif_p))
    data = pd.DataFrame(data).transpose()
    data.columns = ['date','beta','gamma','r0', 'dif','r0_p','dif_p']
    data.to_csv(dir_path +'paramPredict.csv')

saveParam('China', 4, 0)
saveParam('Italy', 3, 9)

# Save new results for China size = 4
saveParam(country = 'China', size = 4, t=0, T = 140)


# A few corrections in this code chunk

def predictFuture(country, size,t, T):
    I,R,D,C = load_IR(country,popu_dict[country], t, T)
    beta_list,gamma_list = read_opt(I,country,size)
    r0,dif = parameterPlot(beta_list,gamma_list)
    datasets_X = np.arange(0, len(r0)).reshape([len(r0),1])
    dif_new = paramPredict(datasets_X[-size:], dif[-size:])
    r0_new = paramPredict(datasets_X[-size:], r0[-size:])
    # Modify gamma, commented gamma is the former one
    #gamma = dif_new / (r0_new - 1)
    gamma = dif_new / (1 - r0_new)
    beta = r0_new * gamma
    print(beta, gamma)
    I_t_p,I_t_all = read_I_t(I,country,size)
    predictPlot(I,I_t_p)
    S_t,I_t,R_t = RungeKutta(beta, gamma, popu_dict[country], popu_dict[country] - I[-1] - R[-1], I[-1], R[-1], 0)
    I_p = list(I_t_p)[:len(I)]
    I_p = I_p + list(I_t)
    I_p = np.array(I_p)
    predictPlot(I,I_p)
    
    C_p = list(C)
    ax  = plt.figure(figsize=(13, 8))
    for i in range(len(C), len(I_p)):
        a = C_p[i-1] + (C_p[i-1] - C_p[i-2])*0.97
        C_p.append(a)
    
    dirs = '\\data\\'
    dir_path = os.path.dirname(os.path.abspath('__file__')) + dirs
    dir_path = dir_path+country+'\\'
    date_X = create_assist_date("2020-1-22", "2021-10-01")
    data = []
    for i in range(0, len(I_p)):
        if i < len(I):
            data.append((date_X[i], I[i],R[i],D[i],C[i],I_p[i],C_p[i]))
        else:
            data.append((date_X[i], '', '', '', '', I_p[i],C_p[i]))
    df = pd.DataFrame(data)
    df.columns=['date','I', 'R', 'D', 'C', 'I_p','C_p'] # add R, D, C here
    df.to_csv(dir_path +'predictFuture.csv')

predictFuture('China', 4, 0)
predictFuture('Italy', 3, 9)

# Get new predicted results for China with size = 4
predictFuture(country = 'China', size = 4, t = 0, T = 140)

