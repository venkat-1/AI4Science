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





    # Read Data
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
prop  = csvdata[:,3]
#prop  = csvdata[:,4]
X = csvdata[:,5:]


    # Read CdTe_0.5Se_0.5 Data
ifile2  = open('Outside.csv', "rt")
reader2 = csv.reader(ifile2)
csvdata2=[]
for row2 in reader2:
        csvdata2.append(row2)
ifile2.close()
numrow2=len(csvdata2)
numcol2=len(csvdata2[0])
csvdata2 = np.array(csvdata2).reshape(numrow2,numcol2)
dopant_out = csvdata2[:,0]
CdX_out = csvdata2[:,1]
doping_site_out = csvdata2[:,2]
prop_out  = csvdata2[:,3]
#prop_out  = csvdata2[:,4]
X_out = csvdata2[:,5:21]

n_out = prop_out.size


    # Read Entire Dataset                                                                                                              
ifile3  = open('X.csv', "rt")
reader3 = csv.reader(ifile3)
csvdata3=[]
for row3 in reader3:
        csvdata3.append(row3)
ifile3.close()
numrow3=len(csvdata3)
numcol3=len(csvdata3[0])
csvdata3 = np.array(csvdata3).reshape(numrow3,numcol3)
dopant_all = csvdata3[:,0]
CdX_all = csvdata3[:,1]
doping_site_all = csvdata3[:,2]
X_all = csvdata3[:,3:19]





XX = copy.deepcopy(X)
n = dopant.size
m = np.int(X.size/n)

t = 0.10

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

n_est_all = [100, 200, 500]
max_depth_all = [50, 100, 150]
max_feat_all = [10, 15, m]
min_samp_leaf_all = [1, 3, 5]
min_samp_split_all = [2, 5, 10]

for i in range(0,3):
    for j in range(0,3):
        rfreg_temp = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=n_est_all[i], max_depth=max_depth_all[j], max_features=15, min_samples_leaf=5, min_samples_split=10)
        rfregs_all.append(rfreg_temp)









#times = len(rfregs_all)
times = 1

train_errors = [0.0]*times
test_errors = [0.0]*times
rfregs = list()

n_fold = 10

for i in range(0,times):
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    count = 0
    rfreg = rfregs_all[8]
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv, dop_train_cv, dop_test_cv, sc_train_cv, sc_test_cv, ds_train_cv, ds_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test], dop_train[train], dop_train[test], sc_train[train], sc_train[test], ds_train[train], ds_train[test]

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

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    rfregs.append(rfreg)
i_opt = np.argmin(test_errors)
rfreg_opt = rfregs[i_opt]


rfreg_opt.fit(X_train,Prop_train)
Pred_train = rfreg_opt.predict(X_train)
Pred_test  = rfreg_opt.predict(X_test)
Pred_train_fl = [0.0]*(Pred_train.size)
Pred_test_fl = [0.0]*(Pred_test.size)
for i in range(0,Pred_train.size):
    Pred_train_fl[i] = np.float(Pred_train[i])
for i in range(0,Pred_test.size):
    Pred_test_fl[i] = np.float(Pred_test[i])


np.savetxt('Prop_test.txt', Prop_test_fl)
np.savetxt('Pred_test.txt', Pred_test_fl)








##  Outside Predictions  ##


Pred_out = rfreg_opt.predict(X_out)
Pred_out_fl = [0.0]*Pred_out.size
Prop_out_fl = [0.0]*Pred_out.size
for i in range(0,prop_out.size):
    Prop_out_fl[i] = np.float(prop_out[i])
for i in range(0,prop_out.size):
    Pred_out_fl[i] = np.float(Pred_out[i])

np.savetxt('Pred_out.csv',Pred_out_fl)




Pred_all = rfreg_opt.predict(X_all)
Pred_all_fl = [0.0]*Pred_all.size
for i in range(0,Pred_all.size):
    Pred_all_fl[i] = np.float(Pred_all[i])

np.savetxt('Pred_all.csv',Pred_all_fl)



mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_fl,Pred_test_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_fl,Pred_train_fl)
mse_out = sklearn.metrics.mean_squared_error(Prop_out_fl, Pred_out_fl)


print('rmse_test_prop=', np.sqrt(mse_test_prop))
print('rmse_train_prop=', np.sqrt(mse_train_prop))




mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_fl,Pred_test_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_fl,Pred_train_fl)
mse_CdTeSe = sklearn.metrics.mean_squared_error(Prop_out_fl[0:132],Pred_out_fl[0:132])
mse_CdSeS = sklearn.metrics.mean_squared_error(Prop_out_fl[132:264],Pred_out_fl[132:264])


print('rmse_test_prop=', np.sqrt(mse_test_prop))
print('rmse_train_prop=', np.sqrt(mse_train_prop))
print('rmse_CdTeSe=', np.sqrt(mse_CdTeSe))
print('rmse_CdSeS=', np.sqrt(mse_CdSeS))


imp = rfreg_opt.feature_importances_






percentile = 95


err_up_train   = [0.0]*n_tr
err_down_train = [0.0]*n_tr
preds_fl = [[0.0 for a in range(n_tr)] for b in range (len(rfreg_opt.estimators_))]
z = 0

for pred in rfreg_opt.estimators_:
    preds = pred.predict(X_train)
    for i in range(0,n_tr):
        preds_fl[z][i] = np.float(preds[i])
    z = z+1

pp = [0.0]*len(rfreg_opt.estimators_)
for i in range(n_tr):
    for j in range(0,len(rfreg_opt.estimators_)):
        pp[j] = preds_fl[j][i]
    err_down_train[i] = np.percentile(pp[:], (100 - percentile) / 2. )
    err_up_train[i] = np.percentile(pp[:], 100 - (100 - percentile) / 2.)



err_up_test   = [0.0]*n_te
err_down_test = [0.0]*n_te
preds_fl = [[0.0 for a in range(n_te)] for b in range (len(rfreg_opt.estimators_))]
z = 0

for pred in rfreg_opt.estimators_:
    preds = pred.predict(X_test)
    for i in range(0,n_te):
        preds_fl[z][i] = np.float(preds[i])
    z = z+1

pp = [0.0]*len(rfreg_opt.estimators_)
for i in range(n_te):
    for j in range(0,len(rfreg_opt.estimators_)):
        pp[j] = preds_fl[j][i]
    err_down_test[i] = np.percentile(pp[:], (100 - percentile) / 2. )
    err_up_test[i] = np.percentile(pp[:], 100 - (100 - percentile) / 2.)




err_up_out = [0.0]*n_out
err_down_out = [0.0]*n_out
preds_fl = [[0.0 for a in range(n_out)] for b in range (len(rfreg_opt.estimators_))]
z = 0

for pred in rfreg_opt.estimators_:
    preds = pred.predict(X_out)
    for i in range(0,n_out):
        preds_fl[z][i] = np.float(preds[i])
    z = z+1

pp = [0.0]*len(rfreg_opt.estimators_)
for i in range(n_out):
    for j in range(0,len(rfreg_opt.estimators_)):
        pp[j] = preds_fl[j][i]
    err_down_out[i] = np.percentile(pp[:], (100 - percentile) / 2. )
    err_up_out[i] = np.percentile(pp[:], 100 - (100 - percentile) / 2.)


up_out = [0.0]*n_out
down_out = [0.0]*n_out

for i in range(0,n_out):
    up_out[i] = err_up_out[i] - Pred_out_fl[i]
    down_out[i] = Pred_out_fl[i] - err_down_out[i]







Prop_train_CdTe = [0.0]*len(Prop_train_fl)
err_up_train_CdTe = copy.deepcopy(err_up_train)
err_down_train_CdTe = copy.deepcopy(err_down_train)

Prop_train_CdSe = [0.0]*len(Prop_train_fl)
err_up_train_CdSe = copy.deepcopy(err_up_train)
err_down_train_CdSe = copy.deepcopy(err_down_train)

Prop_train_CdS = [0.0]*len(Prop_train_fl)
err_up_train_CdS = copy.deepcopy(err_up_train)
err_down_train_CdS = copy.deepcopy(err_down_train)

Prop_test_CdTe = [0.0]*len(Prop_test_fl)
err_up_test_CdTe = copy.deepcopy(err_up_test)
err_down_test_CdTe = copy.deepcopy(err_down_test)

Prop_test_CdSe = [0.0]*len(Prop_test_fl)
err_up_test_CdSe = copy.deepcopy(err_up_test)
err_down_test_CdSe = copy.deepcopy(err_down_test)

Prop_test_CdS = [0.0]*len(Prop_test_fl)
err_up_test_CdS = copy.deepcopy(err_up_test)
err_down_test_CdS = copy.deepcopy(err_down_test)


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
        err_up_train_CdTe[aa] = err_up_train[i]
        err_down_train_CdTe[aa] = err_down_train[i]
        aa = aa+1
    if sc_train[i] == 'CdSe':
        Prop_train_CdSe[bb] = Prop_train_fl[i]
        Pred_train_CdSe[bb] = Pred_train_fl[i]
        err_up_train_CdSe[bb] = err_up_train[i]
        err_down_train_CdSe[bb] = err_down_train[i]
        bb = bb+1
    if sc_train[i] == 'CdS':
        Prop_train_CdS[cc] = Prop_train_fl[i]
        Pred_train_CdS[cc] = Pred_train_fl[i]
        err_up_train_CdS[cc] = err_up_train[i]
        err_down_train_CdS[cc] = err_down_train[i]
        cc = cc+1

for i in range(0,Prop_test_fl.size):
    if sc_test[i] == 'CdTe':
        Prop_test_CdTe[dd] = Prop_test_fl[i]
        Pred_test_CdTe[dd] = Pred_test_fl[i]
        err_up_test_CdTe[dd] = err_up_test[i]
        err_down_test_CdTe[dd] = err_down_test[i]
        dd = dd+1
    if sc_test[i] == 'CdSe':
        Prop_test_CdSe[ee] = Prop_test_fl[i]
        Pred_test_CdSe[ee] = Pred_test_fl[i]
        err_up_test_CdSe[ee] = err_up_test[i]
        err_down_test_CdSe[ee] = err_down_test[i]
        ee = ee+1
    if sc_test[i] == 'CdS':
        Prop_test_CdS[ff] = Prop_test_fl[i]
        Pred_test_CdS[ff] = Pred_test_fl[i]
        err_up_test_CdS[ff] = err_up_test[i]
        err_down_test_CdS[ff] = err_down_test[i]
        ff = ff+1



up_train_CdTe = [0.0]*aa
down_train_CdTe = [0.0]*aa
up_test_CdTe = [0.0]*dd
down_test_CdTe = [0.0]*dd

up_train_CdSe = [0.0]*bb
down_train_CdSe = [0.0]*bb
up_test_CdSe = [0.0]*ee
down_test_CdSe = [0.0]*ee

up_train_CdS = [0.0]*cc
down_train_CdS = [0.0]*cc
up_test_CdS = [0.0]*ff
down_test_CdS = [0.0]*ff


for i in range(0,aa):
    up_train_CdTe[i]   = err_up_train_CdTe[i] - Pred_train_CdTe[i]
    down_train_CdTe[i] = Pred_train_CdTe[i] - err_down_train_CdTe[i]
for i in range(0,bb):
    up_train_CdSe[i] = err_up_train_CdSe[i] - Pred_train_CdSe[i]
    down_train_CdSe[i] = Pred_train_CdSe[i] - err_down_train_CdSe[i]
for i in range(0,cc):
    up_train_CdS[i] = err_up_train_CdS[i] - Pred_train_CdS[i]
    down_train_CdS[i] = Pred_train_CdS[i] - err_down_train_CdS[i]



for i in range(0,dd):
    up_test_CdTe[i] = err_up_test_CdTe[i] - Pred_test_CdTe[i]
    down_test_CdTe[i] = Pred_test_CdTe[i] - err_down_test_CdTe[i]
for i in range(0,ee):
    up_test_CdSe[i] = err_up_test_CdSe[i] - Pred_test_CdSe[i]
    down_test_CdSe[i] = Pred_test_CdSe[i] - err_down_test_CdSe[i]
for i in range(0,ff):
    up_test_CdS[i] = err_up_test_CdS[i] - Pred_test_CdS[i]
    down_test_CdS[i] = Pred_test_CdS[i] - err_down_test_CdS[i]




plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.13, bottom=0.12, right=0.96, top=0.96)

#Data = pandas.read_excel('Pred.xlsx','Sheet1')

#a = [-20,-15,-10,-5,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#b = [-20,-15,-10,-5,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

a = [-175,0,125]
b = [-175,0,125]

plt.rc('font', family='Arial narrow')

plt.plot(b, a, c='k', ls='-')


plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)
#plt.ylim([0.0,15.0])
#plt.xlim([0.0,15.0])
plt.ylim([-1.5,4.2])
plt.xlim([-1.5,4.2])

#plt.errorbar(Prop_train_fl[:], Pred_train[:], yerr = [err_up_train[:]-Pred_train[:], Pred_train[:]-err_down_train[:]], c='blue', marker='s', alpha=1.0, markeredgecolor='dimgrey', markersize=10, fmt='o', ecolor='blue', capthick=2, label='Training')
#plt.errorbar(Prop_test_fl[:], Pred_test[:], yerr =  [err_up_test[:]-Pred_test[:], Pred_test[:]-err_down_test[:]], c='orange', marker='s', alpha=0.2, markeredgecolor='dimgrey', markersize=10, fmt='o', ecolor='orange', capthick=2, label='Test')

#plt.scatter(Prop_train_CdTe[:], Pred_train_CdTe[:], c='blue', marker='s',edgecolors='dimgrey', alpha=1.0, label='Training')
#plt.scatter(Prop_test_CdTe[:], Pred_test_CdTe[:], c='orange', marker='s', edgecolors='dimgrey', alpha=0.2, label='Test')

#plt.scatter(Prop_train_CdSe[:], Pred_train_CdSe[:], c='blue', marker='^',edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
#plt.scatter(Prop_test_CdSe[:], Pred_test_CdSe[:], c='orange', marker='^', edgecolors='dimgrey', alpha=0.2, label='_nolegend_')

#plt.scatter(Prop_train_CdS[:], Pred_train_CdS[:], c='blue', marker='*',edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
#plt.scatter(Prop_test_CdS[:], Pred_test_CdS[:], c='orange', marker='*', edgecolors='dimgrey', alpha=0.2, label='_nolegend_')


plt.scatter(Prop_train_CdTe[:], Pred_train_CdTe[:], c='blue', marker='s', s=50, edgecolors='dimgrey', alpha=1.0, label='Training')
plt.scatter(Prop_train_CdSe[:], Pred_train_CdSe[:], c='blue',marker='^', s=75, edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
plt.scatter(Prop_train_CdS[:], Pred_train_CdS[:], c='blue', marker='*', s=100, edgecolors='dimgrey', alpha=1.0, label='_nolegend_')
plt.scatter(Prop_test_CdTe[:], Pred_test_CdTe[:], c='orange', marker='s', s=50, edgecolors='dimgrey', alpha=0.2, label='Test')
plt.scatter(Prop_test_CdSe[:], Pred_test_CdSe[:], c='orange', marker='^', s=75, edgecolors='dimgrey', alpha=0.2, label='_nolegend_')
plt.scatter(Prop_test_CdS[:], Pred_test_CdS[:], c='orange', marker='*', s=100, edgecolors='dimgrey', alpha=0.2, label='_nolegend_')

plt.scatter(Prop_out_fl[0:132], Pred_out_fl[0:132], c='red', marker='h', s=75, edgecolors='dimgrey', alpha=0.2, label='CdTe$_{0.5}$Se$_{0.5}$')
plt.scatter(Prop_out_fl[132:264], Pred_out_fl[132:264], c='green', marker='h', s=75, edgecolors='dimgrey', alpha=0.2, label='CdSe$_{0.5}$S$_{0.5}$')

#plt.errorbar(Prop_train_CdTe[0:aa], Pred_train_CdTe[0:aa], yerr = [up_train_CdTe[0:aa], down_train_CdTe[0:aa]], c='blue', marker='s', alpha=1.0, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='blue', capthick=1, label='Training')
#plt.errorbar(Prop_train_CdSe[0:bb], Pred_train_CdSe[0:bb], yerr = [up_train_CdSe[0:bb], down_train_CdSe[0:bb]], c='blue', marker='^', alpha=1.0, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='blue', capthick=1, label='_nolegend_')
#plt.errorbar(Prop_train_CdS[0:cc], Pred_train_CdS[0:cc], yerr = [up_train_CdS[0:cc], down_train_CdS[0:cc]], c='blue', marker='*', alpha=1.0, markeredgecolor='dimgrey', markersize=12, fmt='o', ecolor='blue', capthick=1, label='_nolegend_')

#plt.errorbar(Prop_test_CdTe[0:dd], Pred_test_CdTe[0:dd], yerr = [up_test_CdTe[0:dd], down_test_CdTe[0:dd]], c='orange', marker='s', alpha=0.2, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='orange', capthick=1, label='Test')
#plt.errorbar(Prop_test_CdSe[0:ee], Pred_test_CdSe[0:ee], yerr = [up_test_CdSe[0:ee], down_test_CdSe[0:ee]], c='orange', marker='^', alpha=0.2, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='orange', capthick=1, label='_nolegend_')
#plt.errorbar(Prop_test_CdS[0:ff], Pred_test_CdS[0:ff], yerr = [up_test_CdS[0:ff], down_test_CdS[0:ff]], c='orange', marker='*', alpha=0.2, markeredgecolor='dimgrey', markersize=12, fmt='o', ecolor='orange', capthick=1, label='_nolegend_')

#plt.errorbar(Prop_out_fl[0:132], Pred_out_fl[0:132], yerr = [up_out[0:132], down_out[0:132]], c='red', marker='h', alpha=0.2, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='red', capthick=1, label='CdTe$_{0.5}$Se$_{0.5}$')
#plt.errorbar(Prop_out_fl[132:264], Pred_out_fl[132:264], yerr = [up_out[132:264], down_out[132:264]], c='green', marker='h', alpha=0.2, markeredgecolor='dimgrey', markersize=8, fmt='o', ecolor='green', capthick=1, label='CdSe$_{0.5}$S$_{0.5}$')


plt.xticks([-1, 0, 1, 2, 3, 4])
plt.yticks([-1, 0, 1, 2, 3, 4])
plt.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':24})
#plt.legend(bbox_to_anchor=(0.6, 0.65), frameon=False, prop={'family':'Arial narrow','size':16})
#plt.plot(b, a, c='k', ls='--')
#plt.plot(b, c, c='k', ls='--')
plt.savefig('plot.eps', dpi=450)
#plt.show()









