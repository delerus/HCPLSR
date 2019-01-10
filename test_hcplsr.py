#!/usr/bin/env python3


"""
This is containing tests for the hcplsr class
"""

import pandas as pd
import numpy as np
import pytest
from pytest import approx
import scipy.io as spio  # Used to extract matlab files
from copy import copy

from hcplsr import Hcplsr




@pytest.fixture()
def pca_scaled_secondorder():
    """
    Downloads and extracts matlabresults from pca scaled secondorder
    """

    train_object = spio.loadmat("Results_matlab/pca/scaled/secondorder/h.mat")
    train_model = train_object['h'][0,0]
    matlab_dict = {}
    for i,k in enumerate(train_model.dtype.names):
        matlab_dict[k] = train_model[i]

    return matlab_dict 

@pytest.fixture()
def xscores_scaled_secondorder():
    """
    Downloads and extracts matlabresults from xscores scaled secondorder
    """
    train_object = spio.loadmat("Results_matlab/xscores/scaled/secondorder/h.mat")
    train_model = train_object['h'][0,0]
    matlab_dict = {}
    for i,k in enumerate(train_model.dtype.fields.keys()):
        matlab_dict[k] = train_model[i]

    return matlab_dict 

@pytest.fixture()
def yscores_scaled_secondorder():
    """
    Downloads and extracts matlabresults from yscores scaled secondorder
    """
    train_object = spio.loadmat("Results_matlab/yscores/scaled/secondorder/h.mat")
    train_model = train_object['h'][0,0]
    matlab_dict = {}
    for i,k in enumerate(train_model.dtype.fields.keys()):
        matlab_dict[k] = train_model[i]

    return matlab_dict 

@pytest.fixture()
def test_data():
    """
    Downloads test data
    """
    xpath = "Results_matlab/test_data/testdataX.xlsx"
    ypath = "Results_matlab/test_data/testdataY.xlsx"

    X_df = pd.read_excel(xpath,header=None)
    Y_df = pd.read_excel(ypath,header=None)
    X = X_df.values
    Y = Y_df.values

    return X,Y


def test_extract_matlab(pca_scaled_secondorder,xscores_scaled_secondorder,yscores_scaled_secondorder):
    """
    Tests that the matlab results are downloaded correctly
    """
    matlab_results1 = pca_scaled_secondorder
    matlab_results2 = xscores_scaled_secondorder
    matlab_results3 = yscores_scaled_secondorder

    assert len(matlab_results1) > 2
    assert len(matlab_results2) > 2
    assert len(matlab_results3) > 2

def test_Data(test_data):
    """
    Tests that the test data is extracted correctly
    """

    X,Y = test_data

    assert X.shape == (150,4)
    assert X[0,0] == 5.1
    assert Y.shape == (150,4)
    assert approx(Y[0,0]) == 48.892942

def test_standardize(test_data,pca_scaled_secondorder):
    
    matlab_results = pca_scaled_secondorder
    X,Y = test_data
    
    hcplsr = Hcplsr(standard_X=True,standard_Y=True,secondorder=True)
    hcplsr.fit(X,Y)


    assert approx(hcplsr.mX_0) == matlab_results['mX'][0]
    assert approx(hcplsr.mY) == matlab_results['mY'][0]
    assert approx(hcplsr.stdY) == matlab_results['stdY'][0]
    assert approx(hcplsr.mX) == matlab_results['mX_2'][0]
    assert approx(hcplsr.stdX) == matlab_results['stdX'][0]




