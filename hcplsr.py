#!/usr/bin/env python3


"""
This is going to be a script for the HCPLSR class
"""

import numpy as np

from copy import copy

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_squared_error,euclidean_distances

from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

class Hcplsr():

    def __init__(self,matrix='T',n_clusters=6,standard_X=True,standard_Y=True,sincos=False,secondorder=False,m=2,n_comps=10,pca_standard=False):
        """
        Docstring for the HCPLSR

        """

        self.matrix = matrix
        self.n_clusters=n_clusters
        self.m=m

        self.standard_X = standard_X
        self.standard_Y = standard_Y

        self.sincos=sincos
        self.secondorder=secondorder

        self.n_comps = n_comps

        self.pca_standard = pca_standard


    def fit(self,X,Y):

        self.X_org = copy(X)
        self.Y_org = copy(Y)


        if self.sincos or self.secondorder:
            X = self.add_terms(X,fit=True)

        self.mX = np.mean(X,axis=0)
        self.stdX = np.std(X,axis=0,ddof=1)
        self.mY = np.mean(Y,axis=0)
        self.stdY = np.std(Y,axis=0,ddof=1)

        if self.standard_X:
            for col in range(X.shape[1]):
                X[:,col] = (X[:,col]-self.mX[col])/self.stdX[col]
        if self.standard_Y:
            for col in range(Y.shape[1]):
                Y[:,col] = (Y[:,col]-self.mY[col])/self.stdY[col]

        #Global PLSR model starts here
        self.n_comps = min(self.n_comps,X.shape[1])

        self.gplsr = PLSRegression(n_components=self.n_comps,scale=False)
        self.gplsr.fit(X,Y)
        
        mse = self.calc_MSE(X,Y,self.gplsr)
        self.opt_PCs = self.Opt_PCs(mse)

        self.ypred = self.gplsr.predict(X)
        self.R2 = r2_score(Y,self.ypred)

        #Hiearichal model starts here.

        #Choosing matrix
        if self.matrix == 'Xscores':
            self.matr = self.gplsr.x_scores_[:,:self.opt_PCs]
        elif self.matrix == 'Yscores':
            self.matr = self.gplsr.y_scores_[:,:self.opt_PCs]
        elif self.matrix == 'T':
            #The data is not standarized before pca is used in matlab file.
            #But I will use standarized data for now.

            if self.secondorder:

                pf = PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
                X = np.copy(self.X_org)
                X = X[:,:]-self.mX_0
                X_temp = pf.fit_transform(X)
                X = np.concatenate((X_temp,X**2),axis=1)

            if self.pca_standard:
                for col in range(X.shape[1]):
                    X[:,col] = (X[:,col] - self.mX[col])/self.stdX[col]


            self.pca = PCA()
            self.T = self.pca.fit_transform(X)
            var = self.pca.explained_variance_

            self.pca_PCs = 1
            for i in range(1,len(var)):
                diff = abs(var[i]-var[i-1])
                if diff > 0.01*var[0] and self.pca_PCs < len(var)-1:
                    self.pca_PCs=i+1
            self.matr = self.T[:,:self.pca_PCs]

        #Clustering

        fk = FuzzyKMeans(k=self.n_clusters, m=self.m).fit(self.matr)
        self.cluster_centers = fk.cluster_centers_
        D,U = self.fuzzy_predictor(self.matr)
        self.labels = np.argmax(U, axis=1)

        #Removing small clusters
        empty_class,small_class,small_rows = self.small_clusters()
        del_class = copy(empty_class)
        del_class.extend(small_class)
        class_ok = np.setdiff1d([i for i in range(self.n_clusters)],del_class)

        if del_class:
            if del_class == self.n_clusters:
                self.labels = np.ones(self.matr.shape[0])
                print('only one cluster, only gplsr created')
            else:
                self.cluster_centers = self.cluster_centers[class_ok,:]
                D,U = self.fuzzy_predictor(self.matr)
                self.labels = np.argmax(U, axis=1)
                self.labels[small_rows] = -1

                self.n_clusters = max(self.labels+1)
                print('n_clusters reduced to:',self.n_clusters)

        self.find_outlierlimits(D)

        # Local modelling
        
        self.optPCgr = []
        self.lplsr = []
        self.ypredgr = np.zeros_like(Y)

        for i in range(self.n_clusters):
            sample_index = self.labels==i
            Xgr = X[sample_index,:]
            Ygr = Y[sample_index,:]

            if self.secondorder:
                #Seperate main and cross effects
                maineff = np.append(np.ones([Xgr.shape[0],1]),Xgr[:,:self.X_org.shape[1]],axis=1)
                crosseff = Xgr[:,self.X_org.shape[1]:]
                #Model: Z=maineff*Bmain_cross+D

                Bmain_cross=np.dot(np.linalg.pinv(maineff),crosseff)
                D=crosseff-np.dot(maineff,Bmain_cross)
                Xgr=np.append(maineff[:,2:],D,axis=1)

            self.optPCgr.append(min(self.opt_PCs,Xgr.shape[0])) # This should be changed to optimal
            self.lplsr.append(PLSRegression(n_components=self.optPCgr[i],scale=False).fit(Xgr,Ygr))
            self.ypredgr[sample_index,:] = self.lplsr[i].predict(Xgr)

        if small_rows:
            self.ypredgr[small_rows,:] = self.ypred[small_rows]


    



                


    def add_terms(self,X,fit=False):

        if self.sincos:
            pass

        elif self.secondorder:
            if fit:
                self.mX_0 = np.mean(X,axis=0)
                self.pf = PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
                self.pf.fit(X[:,:]-self.mX_0)

            X = X[:,:]-self.mX_0
            X_temp = self.pf.transform(X) 
            X = np.concatenate((X_temp,X**2),axis=1) #Second order terms added later because of testing purposes with matlab.
        return X

    @staticmethod
    def calc_MSE(X,Y,plsr):

        PCs = plsr.n_components
        MSE = []
        for i in range(1,PCs):
            coef = np.dot(plsr.x_rotations_[:,:i],plsr.y_rotations_[:,:i].T)
            Ypred = np.dot(X,coef)
            MSE.append(mean_squared_error(Y,Ypred))
        return MSE

    @staticmethod
    def Opt_PCs(mse):
        optPCs = 1
        for i in range(2,len(mse)):
            diff = abs(mse[i]-mse[i-1])
            if diff > 0.01*abs(mse[1]):
                optPCs = i
        return optPCs

    def fuzzy_predictor(self,X):
        """

        returns:
        D ~ Distance matrix
        U ~ Fuzzy labels
        """

        D = euclidean_distances(X,self.cluster_centers, squared=True)

        U = 1.0 / D
        U **= 1.0 / (self.m-1)
        U /= np.sum(U, axis=1)[:, np.newaxis]

        return D,U

    def small_clusters(self):

        empty_class = []
        small_class = []
        small_rows = []

        un_labels = np.unique(self.labels)
        for i in range(self.n_clusters):
            if i not in un_labels:
                empty_class.append(i)
            elif sum(self.labels==i) < 10:
                small_class.append(i)
                small_rows.extend(list(np.where(self.labels==i)[0]))
        return empty_class,small_class,small_rows

    def find_outlierlimits(self,D):
        self.outlierlimits = []
        for i in range(self.n_clusters):
            self.outlierlimits.append(np.max(D[self.labels==i,i]))
        return np.array(self.outlierlimits)












