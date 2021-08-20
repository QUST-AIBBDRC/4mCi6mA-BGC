import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.manifold import SpectralEmbedding 


def SE(data,n_components=10):
    embedding = SpectralEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed

data_=pd.read_csv(r'DNA_space.csv')
data=np.array(data_)
data=data[:,2:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu1=scale(data)
data_2=SE(shu1,n_components=10)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('SE.csv')