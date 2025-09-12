# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

"""
Created on July, 2025

@author: wangc, mandd
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import pairwise_distances

import logging
logger = logging.getLogger(__name__)

from .kernel_two_sample_test import kernel_two_sample_test


def getL(loc,window,array,type):
    """
      Method designed to extract a portion of a time series

      Args:
        loc: int, reference location in the time series for the selected portion
        window: np.array, size of the selected portion of the time series
        array: np.array, original time series
        type: string, type of the selected portion of the time series (rear or front)
      Return:
        timeseriesPortion: np array, size of the selected portion of the time series
    """
    if type not in ['rear','front']:
        logger.error('getL method: Specificed type to select portion of time series (' + str (type) + ') is not allowed. Type can be either rear or front.')
    
    if isinstance(array, np.ndarray) or isinstance(array, pd.Series):
        if type=='rear':
            l = array[loc:loc+window]
        elif type=='front':
            l = array[loc-window:loc]
        else:
            logger.error('error')
    else:
        logger.error('getL method: Provided time series has incorrect data type.')
    
    timeseriesPortion = np.asarray(l)
    return timeseriesPortion


def omega(window,array,Nsamples):
    """
      Method designed to extract set of portions of a time series

      Args:
        window: np.array, size of the selected portion of the time series
        array: np.array, original time series
        Nsamples: int, number of portions to be selected
      Return:
        omegaSet: np.array, set of Nsamples portions of a time series
    """
    size = len(array)
    if isinstance(array, np.ndarray) or isinstance(array, list):
        omegaSet=np.zeros([Nsamples,window])
        start = np.random.randint(low=0, high=(size-window), size=Nsamples)
        for i in range(Nsamples):
            omegaSet[i,:] = array[start[i]:start[i]+window]
    elif isinstance(array, pd.Series):
        omegaSet = []
        # pd.Series with Date as index, the length with window is not consistent
        # omegaSet=np.zeros([Nsamples,len(array[:array.index[0]+window])])
        high = len(array[:array.index[-1]-window])
        start = np.random.randint(low=0, high=high, size=Nsamples)
        for i in range(Nsamples):
            loc = array.index[start[i]]
            sample = array[loc:loc+window]
            omegaSet.append(np.asarray(sample))
    return omegaSet

def choice(array,Nsamples):
    """
      Method designed to randomly choose Nsamples out of an array (replace is set to True)

      Args:
        array: np.array, array of values to be sampled
        Nsamples: int, number of elements of array to be selected
      Return:
        sampled: np.array, set of Nsamples elements randomly chosen from array
    """
    sampled = np.random.choice(array, size=Nsamples, replace=True)
    return sampled

def t_score(array_front, array_rear):
    """
      Method designed to calculate the statistical difference between two arrays using t-test

      Args:
        array_front: np.array, first array
        array_rear: np.array, second array
      Return:
        tscore: float, outcome of t-test
    """

    if array_front.size == array_rear.size:
        n = float(array_front.size)

        mean_front = np.mean(array_front)
        var_front  = np.var(array_front)

        mean_rear = np.mean(array_rear)
        var_rear  = np.var(array_rear)

        tscore = (mean_front-mean_rear)/math.sqrt((var_front+var_rear)/n)
    else:
        logger.error("t_score error: the two provided array have different sizes.")
        tscore = None
    return tscore

def MMD_test(test_array, omegaSet, iterations, alphaTest, alphaOmegaset, printFlag=True):
    """
      Method designed to calculate the statistical difference between an array and a set of arrays 
      using the Maximum Mean Discrepancy testing method

      Args:
        test_array: np.array, array of values to be tested against omegaSet
        omegaSet: np.ndarray or list, population of arrays
        iterations: int, number of iterartions required to compute MMD^2_u
        alphaTest: float, acceptance value for hypothesis testing (single array testing)
        alphaOmegaset: float, acceptance value for hypothesis testing (omegaSet testing)
        printFlag: bool, flag to plot MMD distribution
      Return:
        p_value: float, p-value obtained from MMD testing
        testOutput: bool, logical outcome of MMD testing
    """

    if isinstance(omegaSet, np.ndarray):
        omegaSetDim  = omegaSet.ndim
    elif isinstance(omegaSet, list):
        if isinstance(omegaSet[0], np.ndarray):
            omegaSetDim = len(omegaSet)
        else:
            omegaSetDim = 1

    if omegaSetDim==1:
        sigma2 = np.median(pairwise_distances(test_array.reshape(-1, 1), omegaSet.reshape(-1, 1), metric='euclidean'))**2
        mmd2u, mmd2u_null, p_value = kernel_two_sample_test(test_array, omegaSet,
                                                            kernel_function='rbf',
                                                            iterations=iterations,
                                                            gamma=1.0/sigma2,
                                                            verbose=False)
        # print("Null hypothesis: the two sub-series are generated from the same distribution")
        if p_value < alphaTest:
            nullHypothesis = False
            testOutput = True
            logger.info(f'The p-value is: {round(p_value, 5)}, which is less than the significant level: {alphaTest}')
            # print("The null hypothesis got rejected")
            # print('The two sub-series are generated from two different distributions')
        else:
            nullHypothesis = True
            testOutput = False
            logger.info(f'The p-value is: {round(p_value, 5)}, which is larger than the significant level: {alphaTest}')
            # print("The null hypothesis got accepted")
            # print('The two sub-series are generated from the same distributions')
        return p_value, testOutput
    else:
        omegaSetSize = len(omegaSet)
        mmd2u_arr    = np.zeros(omegaSetSize)
        mmd2u_null_agg = []
        p_value_arr  = np.zeros(omegaSetSize)
        if isinstance(omegaSet, np.ndarray):
            sigma2 = np.std(omegaSet.ravel())**2
        elif isinstance(omegaSet, list):
            ravelData = []
            for data in omegaSet:
                ravelData.extend(data)
            sigma2 = np.std(ravelData)**2
        for i in range(omegaSetSize):
            # sigma2 = np.median(pairwise_distances(test_array.reshape(-1, 1), omegaSet[i,:].reshape(-1, 1), metric='euclidean'))**2
            if isinstance(omegaSet, np.ndarray):
                data = omegaSet[i, :]
            else:
                data = omegaSet[i]
            mmd2u_arr[i], mmd2u_null, p_value_arr[i] = kernel_two_sample_test(test_array, data,
                                                                              kernel_function='rbf',
                                                                              iterations=iterations,
                                                                              gamma=1.0/sigma2,
                                                                              verbose=False)
            mmd2u_null_agg.extend(list(mmd2u_null))
        # if printFlag:
        #     fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        #     ax1.hist(mmd2u_arr,   bins=30, alpha=0.35, density=True, color='green', label='mmd2u')
        #     ax1.legend()

        #     ax2.hist(p_value_arr, bins=30, alpha=0.35, density=True, color='red', label='p_value')
        #     ax2.legend()

        #     ax3.scatter(mmd2u_arr, p_value_arr, marker='.')
        #     ax3.set_xlabel('mmd2u')
        #     ax3.set_ylabel('p_value')

        #     plt.show()


        # the number of rejections for the null hypothesis in two sample testing
        count = (p_value_arr<alphaTest).sum()
        # print('Number of rejection ratio for the null hypothesis in two sample testing for two given sub-series is:', count/omegaSetSize)
        # probability two signals are from the same distribution
        p_value = 1. - count/omegaSetSize
        # Null hypothesis: the sub-series are generated from the normal conditions of full time-series
        # if the probability lower than alphaOmegaset, reject the null hypothesis
        # print("Null hypothesis: the sub-series are generated from the normal conditions of full time-series")

        if p_value < alphaOmegaset:
            nullHypothesis = False
            testOutput = True
            logger.info(f"The probability of the sub-series are generated from the normal conditions of full time-series is {p_value}, which is smaller than given test value {alphaOmegaset}")
            # print("The null hypothesis got rejected")
            # print('The sub-series is correlated to the anomaly events')
        else:
            nullHypothesis = True
            testOutput = False
            logger.info(f"The probability of the sub-series are generated from the normal conditions of full time-series is {p_value}, which is greater than given test value {alphaOmegaset}")
            # print("The null hypothesis got accepted")
            # print('The sub-series is not correlated to the anomaly events')

        if printFlag:
            fig, ax = plt.subplots()
            bin1 =30
            ax.hist(mmd2u_null_agg, bins=bin1, alpha=1.0, density=True, color='red', label='$MMD^2_u$ Null Distribution', stacked=True)
            bin2 = int((max(mmd2u_arr) - min(mmd2u_arr))/(max(mmd2u_null_agg) - min(mmd2u_null_agg))*bin1)
            ax.hist(mmd2u_arr, bins=bin2, alpha=0.5, density=True, color='blue'  , label='$MMD^2_u$ True Distribution', stacked=True)
            ax.legend()
            ax.set_xlabel('$MMD^2_u$')
            ax.set_ylabel('$p(MMD^2_u)$')
            ax.set_title('$MMD^2_u$: null-distribution and true distribution: with $p$-value=%s'
          % round(p_value,5))
            plt.show()
        return p_value, testOutput

def event2TStest(E_loc, TS, iterations, alphaTest, alphaOmegaset, windowSize, omegaSize, returnMinPval=False):
    """
      Method designed to assess temporal correlation of an event E and a time series TS

      Args:
        E_loc: int, temporal location of event E
        TS: np.array, univariate time series
        iterations: int, number of iterartions required to compute MMD^2_u
        alphaTest: float, acceptance value for hypothesis testing (single array testing)
        alphaOmegaset: float, acceptance value for hypothesis testing (omegaSet testing)
        windowSize: int, size of time window prior and after event occurence
        omegaSize: int, number of samples from time series TS
        returnMinPval: bool, flag that indictae whether to return p-value of MMD assessment
      Return:
        relation: string, outcome of the event to timeseries temporal analysis
        minPval: float, p-value of MMD assessment
    """

    omegaSet = omega(windowSize,TS,omegaSize)
    L_front  = getL(E_loc,windowSize,TS,'front')
    L_rear   = getL(E_loc,windowSize,TS,'rear')
    L_front_rear = np.concatenate((L_front, L_rear), axis=None)
    minPval = 1.0

    logger.info('=============================================')
    # print('MMD Testing')
    logger.info("Computing statitics for Lfront vs. omega")
    p_val_f , test_out_f = MMD_test(L_front, omegaSet, iterations, alphaTest, alphaOmegaset, False)
    logger.info('Correlated?', test_out_f, 'p value:', p_val_f)
    if p_val_f < minPval:
        minPval = p_val_f

    logger.info("\nComputing statistics for Lrear vs. omega")
    p_val_r , test_out_r = MMD_test(L_rear, omegaSet, iterations, alphaTest, alphaOmegaset, False)
    logger.info('Correlated?', test_out_r, 'p value:', p_val_r)
    if p_val_r < minPval:
        minPval = p_val_r

    logger.info("\nComputing statitics for Lfront vs. Lrear")
    p_val_3 , test_out_3 = MMD_test(L_rear, L_front, iterations, alphaTest, alphaOmegaset, False)
    logger.info('Correlated?', test_out_3, 'p value:', p_val_3)

    logger.info("\nComputing statitics for Lfront u Lrear vs. omega")
    p_val_4, test_out_4 = MMD_test(L_front_rear, omegaSet, iterations, alphaTest, alphaOmegaset, False)
    logger.info('Correlated?', test_out_4, 'p value:', p_val_4)
    if p_val_4 < minPval:
        minPval = p_val_4

    # tscore = t_score(L_front, L_rear)

    relation = None

    ###############################################
    #   Old detection logic
    #
    # if test_out_3==False or test_out_4==False:
    #     # add more output info here [congjian]
    #     print('S and E are uncorrelated')
    #     relation = 'S and E are uncorrelated'
    # elif test_out_r==True  and test_out_f==False:
    #     print('E --> S')
    #     relation = 'E --> S'
    # elif test_out_r==False and test_out_f==True:
    #     print('S --> E')
    #     relation = 'S --> E'
    # elif test_out_r==True  and test_out_f==True:
    #     print('S;E')
    #     relation = 'S;E'
    #     '''
    #     if p_val_f>p_val_r:
    #           print('S -->* E')
    #     else:
    #           print('E -->* S')
    #     '''
    # else:
    #     print('S and E are undecided')
    #     relation = 'S and E are undecided'

    #####################################
    # New Detection Logic
    #
    if   test_out_r==True  and test_out_f==False:
        relation = 'E --> S'
    elif test_out_r==False and test_out_f==True:
        relation = 'S --> E'
    elif test_out_r==True  and test_out_f==True:
        #relation = 'S;E'
        if p_val_f<p_val_r:
            relation = 'S -->* E'
        else:
            relation = 'E -->* S'

    else:
        if test_out_4:
            relation = 'S;E'
        else:
            if test_out_3:
                # S and E may be correlated
                relation = 'S?E'
            else:
                # S and E are uncorrelated
                relation = 'S!E'
    logger.info('Identified relation: ', relation)
    if returnMinPval:
        return relation, minPval
    else:
        return relation

