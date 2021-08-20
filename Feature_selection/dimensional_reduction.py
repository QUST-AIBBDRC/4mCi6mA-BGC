import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.feature_selection import SelectFromModel
from lightning.classification import CDClassifier 
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoLarsCV,LassoLars
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression


def mutual_mutual(data,label,k=300):
    model_mutual= SelectKBest(mutual_info_classif, k=k)
    new_data=model_mutual.fit_transform(data, label)
    return new_data
	

     
def lassodimension(data,label,alpha=np.array([0.01,0.05,0.1])):
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data,label)
    mask = x_lasso.coef_ != 0 
    new_data = data[:,mask] 
    return new_data,mask 