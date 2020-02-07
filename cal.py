#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:51:27 2020

@author: greynious
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
begin_day = 1
day_num = 25


def get_feature(degree,day_range):
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(day_range)
    #print(x_poly)
    suspect = data[3]#.reshape(-1,1)
    new = np.zeros((3,3))
    #print(new)
    for i in range(3,day_range.size):
        #print(x_poly[i])
        #print(suspect[i])
        #print(suspect[i-3:i])
        new = np.vstack((new,suspect[i-3:i]))
        #print(new)
        #print("------------------------------")
    comb = np.hstack((x_poly,new))
    return comb

def predict_by_suspect(clf,data,rown,degree,color):
    comb = get_feature(degree,data[0].reshape(-1,1))
    #print(comb)
    clf.fit(comb,data[rown].reshape(-1,1))
    day_range=np.linspace(begin_day,begin_day + day_num - 1,day_num) # [1,2,...,day_num]
    result = clf.predict(get_feature(degree,day_range.reshape(-1,1)))
    print("                            ",np.around(result[-1],decimals=1))
    plt.plot(day_range,result,color=color,linestyle='--',marker='.',label='predict')
    #print(clf.coef_)
    
def predict_new(clf,data,rown,degree,color):
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(data[0].reshape(-1,1))
    #print(x_poly)
    clf.fit(x_poly,data[rown].reshape(-1,1))
    day_range=np.linspace(begin_day,begin_day + day_num - 1,day_num) # [1,2,...,day_num]
    result = clf.predict(poly_reg.fit_transform(day_range.reshape(-1,1)))
    print("                            ",np.around(result[-1],decimals=1))
    plt.plot(day_range,result,color=color,linestyle='--',marker='.',label='predict')
    
def predict(clf,data):
    rown=2
    plt.figure()
    predict_new(clf,data,rown,4,'red')
    predict_new(clf,data,rown,5,'blue')
    predict_new(clf,data,rown,6,'fuchsia')
    predict_new(clf,data,rown,7,'orange')
    predict_by_suspect(clf,data,2,4,'cyan')
    predict_by_suspect(clf,data,2,5,'orange')
    plt.plot(np.arange(1,len(data[0])+1),data[rown],color='g',linestyle='--',marker='o',label='true') # 画出真实值
    plt.title("2019-nCov")
    plt.legend()
    plt.show()
    
def read_data():
    with open("data.csv","r") as f:
        data = np.loadtxt(f,delimiter = ",")
        data = data[...,:] # 对数据切片，回溯前几天的预测值，对比准确性时用到
        print()
        return data
        
if __name__ == "__main__":
    print()
    np.set_printoptions(suppress=True)
    data =read_data()
    clf = LinearRegression()
    predict(clf,data)
    