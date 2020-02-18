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
predict_day = 3
flag = False #用于控制是否按疑似人数进行预测


## 生成包含疑似人数的特征
def get_feature_by_suspect(degree,day_range):
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(day_range)
    suspect = data[3]
    new = np.zeros((3,3))
    for i in range(3,day_range.size):
        new = np.vstack((new,suspect[i-3:i]))
    comb = np.hstack((x_poly,new))
    return comb

## 根据疑似人数来生成预测
def predict_by_suspect(clf,data,rown,degree,color,label):
    comb = get_feature_by_suspect(degree,data[0].reshape(-1,1))
    #print(comb)
    clf.fit(comb,data[rown].reshape(-1,1))
    day_range=np.linspace(begin_day,begin_day + day_num - 1,day_num) # [1,2,...,day_num]
    result = clf.predict(get_feature_by_suspect(degree,day_range.reshape(-1,1)))
    print("                            ",np.around(result[-1],decimals=1))
    plt.plot(day_range,result,color=color,linestyle='--',marker='.',label=label)
    #print(clf.coef_)


## 回归的具体实现
def predict_new(clf,data,rown,degree,color,label):
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(data[0].reshape(-1,1))
    #print(x_poly)
    clf.fit(x_poly,data[rown].reshape(-1,1))
    day_range=np.linspace(begin_day,begin_day + day_num - 1,day_num) # [1,2,...,day_num]
    result = clf.predict(poly_reg.fit_transform(day_range.reshape(-1,1)))
    print("                            ",np.around(result[-1],decimals=1))
    plt.plot(day_range,result,color=color,linestyle='--',marker=',',label=label)


## 进行预测
def predict(clf,data):
    rown = 6 # 1代表对武汉的确诊人数进行回归，2代表对总确诊人数进行回归,3代表对疑似人数进行回归，以此类推
    plt.figure()
    
    predict_new(clf,data,rown,4,'green','4')
    predict_new(clf,data,rown,5,'blue','5')
    predict_new(clf,data,rown,6,'fuchsia','6')
    predict_new(clf,data,rown,7,'orange','7')
    
    if rown == 2 and predict_day == 1 and flag == True:
        predict_by_suspect(clf,data,2,4,'cyan','4+suspect')
        predict_by_suspect(clf,data,2,5,'m','5+suspect')
    #plt.plot(data[0],data[rown],color='red',linestyle=':',marker='.',label='true') # 画出真实值
    plt.scatter(data[0],data[rown],c='r',marker='o')# 画出真实值
    plt.title("COVID-19")
    plt.legend()
    plt.show()

## 读取数据
def read_data():
    global day_num
    with open("NCPdata.csv","r") as f:
        data = np.loadtxt(f,delimiter = ",",comments='#',skiprows = 1)
        data = data[...,:] # 对数据切片，回溯前几天的预测值，对比准确性时用到
        day_num = data[-1][0] + predict_day
        data = data.T
        return data
        
if __name__ == "__main__":
    print()
    np.set_printoptions(suppress=True)
    data =read_data()
    clf = LinearRegression()
    predict(clf,data)
    