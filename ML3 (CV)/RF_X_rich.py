from __future__ import print_function
import numpy as np    
import csv
import copy
import random
#import mlpy
import matplotlib.pyplot as plt
#from mlpy import KernelRidge
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import pandas





##  Read Data  ##

ifile  = open('Data.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
dopant = csvdata[:,0]
CdX = csvdata[:,1]
doping_site = csvdata[:,2]
#prop  = csvdata[:,3]  ## Cd-rich Delta_H
#prop  = csvdata[:,4]  ## Mod. Delta_H
prop  = csvdata[:,5]  ## X-rich Delta_H
X = csvdata[:,6:]




##  Train-Test Split  ##

XX = copy.deepcopy(X)
n = dopant.size
m = np.int(X.size/n)

t = 0.20

X_train, X_test, Prop_train, Prop_test, dop_train, dop_test, sc_train, sc_test, ds_train, ds_test = train_test_split(XX, prop, dopant, CdX, doping_site, test_size=t)

n_tr = Prop_train.size
n_te = Prop_test.size


Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    
X_train_fl = [[0.0 for a in range(m)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,m):
        X_train_fl[i][j] = np.float(X_train[i][j])

X_test_fl = [[0.0 for a in range(m)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,m):
        X_test_fl[i][j] = np.float(X_test[i][j])






  ##  Set of RF Model Parameters  ##


rfregs_all = list()

#n_est_all = [50, 100, 200, 300, 500]
#max_depth_all = [30, 75, 100, 150, 200]

n_est_all = [50, 100, 200]
max_depth_all = [40, 70, 100]
max_feat_all = [10, 15, m]
min_samp_leaf_all = [1, 3, 5]
min_samp_split_all = [2, 5, 10]

for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
#                for h in range(0,3):
#        rfreg_temp = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=n_est_all[i], max_depth=max_depth_all[j], max_features=15, min_samples_leaf=5, min_samples_split=10)
#        rfreg_temp = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=100, max_depth=70, min_samples_leaf=min_samp_leaf_all[i], min_samples_split=min_samp_split_all[j], max_features=15)
#                    rfreg_temp = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=n_est_all[i], max_depth=max_depth_all[j], min_samples_leaf=min_samp_leaf_all[k], min_samples_split=min_samp_split_all[l], max_features=max_feat_all[h])
                rfreg_temp = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=n_est_all[i], max_depth=max_depth_all[j], min_samples_leaf=min_samp_leaf_all[k], min_samples_split=min_samp_split_all[l])
                rfregs_all.append(rfreg_temp)







##  Train Random Forest Model  ##


times = len(rfregs_all)

train_errors = [0.0]*times
test_errors = [0.0]*times


##  Cross-validation  ##

n_fold = 5

for i in range(0,times):
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    count = 0
    rfreg = rfregs_all[i]
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]

        rfreg.fit(X_train_cv,Prop_train_cv)
        Prop_pred_train_cv = rfreg.predict(X_train_cv)
        Prop_pred_test_cv  = rfreg.predict(X_test_cv)

        n_tr_cv = Prop_train_cv.size
        n_te_cv = Prop_test_cv.size
        Prop_test_cv_fl = [0.0]*n_te_cv
        Prop_pred_test_cv_fl = [0.0]*n_te_cv
        Prop_train_cv_fl = [0.0]*n_tr_cv
        Prop_pred_train_cv_fl = [0.0]*n_tr_cv

        for a in range(0,n_tr_cv):
            Prop_train_cv_fl[a] = np.float(Prop_train_cv[a])
            Prop_pred_train_cv_fl[a] = np.float(Prop_pred_train_cv[a])
        for a in range(0,n_te_cv):
            Prop_test_cv_fl[a] = np.float(Prop_test_cv[a])
            Prop_pred_test_cv_fl[a] = np.float(Prop_pred_test_cv[a])

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl,Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl,Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
i_opt = np.argmin(test_errors)
rfreg_opt = rfregs_all[i_opt]


rfreg_opt.fit(X_train,Prop_train)
Pred_train = rfreg_opt.predict(X_train)
Pred_test  = rfreg_opt.predict(X_test)
Pred_train_fl = [0.0]*(Pred_train.size)
Pred_test_fl = [0.0]*(Pred_test.size)
for i in range(0,Pred_train.size):
    Pred_train_fl[i] = np.float(Pred_train[i])
for i in range(0,Pred_test.size):
    Pred_test_fl[i] = np.float(Pred_test[i])


#np.savetxt('Prop_test.txt', Prop_test_fl)
#np.savetxt('Pred_test.txt', Pred_test_fl)











##  Predicted Data by Type of CdX Compound  ##


Prop_train_CdTe = [0.0]*len(Prop_train_fl)
Prop_train_CdSe = [0.0]*len(Prop_train_fl)
Prop_train_CdS = [0.0]*len(Prop_train_fl)
Prop_test_CdTe = [0.0]*len(Prop_test_fl)
Prop_test_CdSe = [0.0]*len(Prop_test_fl)
Prop_test_CdS = [0.0]*len(Prop_test_fl)

Pred_train_CdTe = [0.0]*len(Pred_train_fl)
Pred_train_CdSe = [0.0]*len(Pred_train_fl)
Pred_train_CdS = [0.0]*len(Pred_train_fl)
Pred_test_CdTe = [0.0]*len(Pred_test_fl)
Pred_test_CdSe = [0.0]*len(Pred_test_fl)
Pred_test_CdS = [0.0]*len(Pred_test_fl)



aa = 0
bb = 0
cc = 0
dd = 0
ee = 0
ff = 0
gg = 0
hh = 0
ii = 0
jj = 0
kk = 0
ll = 0

for i in range(0,Prop_train_fl.size):
    if sc_train[i] == 'CdTe':
        Prop_train_CdTe[aa] = Prop_train_fl[i]
        Pred_train_CdTe[aa] = Pred_train_fl[i]
        aa = aa+1
    if sc_train[i] == 'CdSe':
        Prop_train_CdSe[bb] = Prop_train_fl[i]
        Pred_train_CdSe[bb] = Pred_train_fl[i]
        bb = bb+1
    if sc_train[i] == 'CdS':
        Prop_train_CdS[cc] = Prop_train_fl[i]
        Pred_train_CdS[cc] = Pred_train_fl[i]
        cc = cc+1

for i in range(0,Prop_test_fl.size):
    if sc_test[i] == 'CdTe':
        Prop_test_CdTe[dd] = Prop_test_fl[i]
        Pred_test_CdTe[dd] = Pred_test_fl[i]
        dd = dd+1
    if sc_test[i] == 'CdSe':
        Prop_test_CdSe[ee] = Prop_test_fl[i]
        Pred_test_CdSe[ee] = Pred_test_fl[i]
        ee = ee+1
    if sc_test[i] == 'CdS':
        Prop_test_CdS[ff] = Prop_test_fl[i]
        Pred_test_CdS[ff] = Pred_test_fl[i]
        ff = ff+1






##  Calculate Prediction RMSE  ##


rmse_test_prop = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_test_fl, Pred_test_fl) )
rmse_train_prop = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_train_fl, Pred_train_fl) )
print('rmse_test_prop=', rmse_test_prop)
print('rmse_train_prop=', rmse_train_prop)

rmse_test_CdTe = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_test_CdTe[0:dd], Pred_test_CdTe[0:dd]) )
rmse_train_CdTe = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_train_CdTe[0:aa], Pred_train_CdTe[0:aa]) )
print('rmse_test_CdTe=', rmse_test_CdTe)
print('rmse_train_CdTe=', rmse_train_CdTe)

rmse_test_CdSe = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_test_CdSe[0:ee], Pred_test_CdSe[0:ee]) )
rmse_train_CdSe = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_train_CdSe[0:bb], Pred_train_CdSe[0:bb]) )
print('rmse_test_CdSe=', rmse_test_CdSe)
print('rmse_train_CdSe=', rmse_train_CdSe)

rmse_test_CdS = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_test_CdS[0:ff], Pred_test_CdS[0:ff]) )
rmse_train_CdS = np.sqrt ( sklearn.metrics.mean_squared_error(Prop_train_CdS[0:cc], Pred_train_CdS[0:cc]) )
print('rmse_test_CdS=', rmse_test_CdS)
print('rmse_train_CdS=', rmse_train_CdS)










##  Plot Regression Results  ##


plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.18, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')

plt.title('X-rich Formation Energy (eV)', fontsize=24, pad=12)

a = [-175,0,125]
b = [-175,0,125]
plt.plot(b, a, c='k', ls='-')

plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
plt.ylim([-1.0, 13.0])
plt.xlim([-1.0, 13.0])

plt.scatter(Prop_train_CdTe[:], Pred_train_CdTe[:], c='blue', marker='s', s=100, edgecolors='dimgrey', alpha=1.0, label='Training')
plt.scatter(Prop_train_CdSe[:], Pred_train_CdSe[:], c='blue',marker='^', s=150, edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
plt.scatter(Prop_train_CdS[:], Pred_train_CdS[:], c='blue', marker='*', s=200, edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
plt.scatter(Prop_test_CdTe[:], Pred_test_CdTe[:], c='orange', marker='s', s=100, edgecolors='dimgrey', alpha=0.2, label='Test')
plt.scatter(Prop_test_CdSe[:], Pred_test_CdSe[:], c='orange', marker='^', s=150, edgecolors='dimgrey', alpha=0.2, label='_nolegend_')
plt.scatter(Prop_test_CdS[:], Pred_test_CdS[:], c='orange', marker='*', s=200, edgecolors='dimgrey', alpha=0.2, label='_nolegend_')

te = '%.2f' % rmse_test_prop
tr = '%.2f' % rmse_train_prop

plt.text(7.5, 2.3, 'Test_rmse = ', c='r', fontsize=16)
plt.text(10.7, 2.3, te, c='r', fontsize=16)
plt.text(11.85, 2.3, 'eV', c='r', fontsize=16)
plt.text(7.3, 1.3, 'Train_rmse = ', c='r', fontsize=16)
plt.text(10.7, 1.3, tr, c='r', fontsize=16)
plt.text(11.85, 1.3, 'eV', c='r', fontsize=16)
#plt.text(7.5, 0.3, 'Out_rmse = ', c='r', fontsize=16)
#plt.text(3.3, 0.3, tr, c='r', fontsize=16)
#plt.text(3.8, 0.3, 'eV', c='r', fontsize=16)

plt.xticks([0, 2, 4, 6, 8, 10, 12])
plt.yticks([0, 2, 4, 6, 8, 10, 12])
plt.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':24})
plt.savefig('plot_X_rich.eps', dpi=450)




