#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 08:55:46 2025

@author: Zach Harrington
"""

### Rayleigh Quotients and Inverse Iteration:
    
    
import numpy as np

# Sample Matrix
A = np.array([
    [3, 1, 1, 0, 0],
    [1, 5, 2, 1, 3],
    [1, 2, 1, 5, 5],
    [0, 1, 5, 4, 4],
    [0, 3, 5, 4, 1]
])

I = np.identity(5)

k = 10 # number of iterations

M,N = np.shape(A)

E,V = np.linalg.eig(A)

eigenvalue = E[0]
eigenvector = V[:,0]

Big_O = E[0] / E[1] 



#### Power Iteration


power_v = np.empty((M,k+1))
power_Lambda = np.zeros(k+1)

power_v[:,0] = np.array([1/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5)]) # initial vector


for i in range(1,k+1):
    w = np.matmul(A, power_v[:,i-1])
    power_v[:,i] = w/np.sqrt(sum(w**2))
    power_Lambda[i] = np.matmul( np.matmul(np.transpose(power_v[:,i]), A), power_v[:,i])
    
    power_value_error = np.abs(eigenvalue - power_Lambda[i])
    power_vector_error = np.sqrt(sum((eigenvector - power_v[:,i]) ** 2 ))
    print(" Power Value Error:", power_value_error, "\n",
          "Power Vector Error:", power_vector_error, "\n") # each value converges



#### Inverse Iteration

Inv_v = np.empty((M,k+1))
Inv_lambda = np.zeros(k+1)
Inv_v[:,0] = np.array([1/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5)]) # initial vector


    # pick mu close to eigenvalue 
mu = -4.2

for i in range(1,k+1):
    w = np.matmul(np.linalg.inv(A - mu*I), Inv_v[:,i-1])
    Inv_v[:,i] = w / np.sqrt(sum(w**2))
    Inv_lambda[i] = np.matmul(np.matmul(np.transpose(Inv_v[:,i]), A), Inv_v[:,i])
    
    Inv_value_error = np.abs(E[3] - Inv_lambda[i])
    Inv_vector_error = np.sqrt(sum((V[:,3] - Inv_v[:,i]) ** 2 ))
    print(" Inverse Value Error:", Inv_value_error, "\n", 
          "Inverse Vector Error:", Inv_vector_error, "\n") # each value converges



#### Rayleigh Quotient Iteration

Ray_v = np.empty((M,k+1))
Ray_lambda = np.zeros(k+1)
Ray_v[:,0] = np.array([1/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5), 0, np.sqrt(2)/np.sqrt(5)]) # initial vector

Ray_lambda[0] = np.matmul(np.matmul(np.transpose(Ray_v[:,0]), A), Ray_v[:,0]) # initial lambda

for i in range(1,k+1):
    w = np.matmul( np.linalg.inv(A - Ray_lambda[i-1] * I), Ray_v[:,i-1])
    Ray_v[:,i] = w / np.sqrt(sum(w**2))
    Ray_lambda[i] = np.matmul( np.matmul( np.transpose(Ray_v[:,i]) , A) , Ray_v[:,i])
    
    Ray_value_error = np.abs(E[1] - Ray_lambda[i])
    Ray_vector_error = np.sqrt(sum((V[:,1] - Ray_v[:,i]) ** 2 ))
    print(" Rayleigh Iteration Value Error:", Ray_value_error, "\n", 
          "Rayleigh Iteration Vector Error:", Ray_vector_error, "\n") # each value converges
    

















    
