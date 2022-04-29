# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:02:22 2022

@author: Johnny
"""
import numpy as np

a = np.array([[1,2,3]])
b = np.array([[4],[5],[6]])
c = 2
d = np.array([[1,1,2]])

print(np.exp(a.dot(b))+[2,2,4],"\n")

dy_deab = 1
dy_dcd = 1
deab_dab = np.exp(32)
dab_da = np.array([[4],[5],[6]])
dab_db = np.array([[1,2,3]])
dcd_dc = np.array([[1,1,2]])
dcd_dd = 2

y = np.array([[1,1,1]])
a = (y*dy_deab*deab_dab).dot(dab_da)
b = np.sum(y*dy_deab*deab_dab*dab_db)
c = np.sum(y*dy_dcd*dcd_dc)
d = y*dy_dcd*dcd_dd

print("a=",a)
print("b=",b)
print("c=",c)
print("d=",d)

