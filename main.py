#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 21:40:24 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;
import matplotlib.image  as img;
import matplotlib.pyplot as plt;
from scipy.signal import convolve2d as conv2d;
def clip(n,N):
    if n <= 0:
        return 0;
    if n >= N-1:
        return N-1;
    return n;

def bilinear(I,x,y,bounds):
    x_min = int(np.floor(x));
    y_min = int(np.floor(y));
    
    x_max = 1 + x_min;
    y_max = 1 + y_min;
    
    u = x - x_min;
    v = y - y_min;
    
    x_min = bounds(x_min,I.shape[0]);
    x_max = bounds(x_max,I.shape[0]);
    y_min = bounds(y_min,I.shape[1]);
    y_max = bounds(y_max,I.shape[1]);
    
    f00 = I[x_min,y_min];
    f01 = I[x_min,y_max];
    f10 = I[x_max,y_min];
    f11 = I[x_max,y_max];
    return (1.0-u)*(1.0-v)*f00+u*(1.0-v)*f10+(1.0-u)*v*f01+u*v*f11;
    
def to_polar(E,R,T,x0,y0,r_min,r_max,ipol,bounds):
    S = np.zeros((R,T)) if len(E.shape) == 2 else np.zeros((R,T,E.shape[2]));
    dr = (np.log(r_max)-np.log(r_min))/R;
    dt = 2.0*np.pi/T;
    for r in range(R):
        for t in range(T):
            tau = r_min*np.exp(dr*r);
            phi = dt*t;
            x = x0 + tau*np.cos(phi);
            y = y0 + tau*np.sin(phi);
            #if 0 <= x < E.shape[0] and 0 <= y < E.shape[1]:
            S[r,t] = ipol(E,x,y,bounds);
    return S;

def to_cart(S,N,x0,y0,r_min,r_max,ipol,bounds):
    E = np.zeros((N,N)) if len(S.shape) == 2 else np.zeros((N,N,S.shape[2]));
    for i in range(N):
        for j in range(N):
            x = i - x0;
            y = j - y0;
            r = S.shape[0]*(np.log(np.sqrt(x**2+y**2))-np.log(r_min))/(np.log(r_max)-np.log(r_min));
            a = np.arctan2(y,x);
            a = a if a > 0 else 2.0*np.pi+a;
            t = 0.5*S.shape[1]*a/np.pi;            
            E[i,j] = ipol(S,r,t,bounds);
    return E;

I = img.imread("lena.jpg")/255.0;

R = 64;
T = 90;

x0 = 0.5+I.shape[0]/2;
y0 = 0.5+I.shape[1]/2;

r_min = 0.1;
r_max = np.linalg.norm(np.array([x0,y0]))

P = to_polar(I,R,T,x0,y0,r_min,r_max,bilinear,clip);

plt.imshow(P)

Q = to_cart(P,I.shape[0],x0,y0,r_min,r_max,bilinear,clip);

plt.imshow(Q)

w = np.outer(np.array([-1.0,0.0,1.0]),np.array([1.0,2.0,1.0])/4.0);

Px = np.zeros(P.shape[:2])
Py = np.zeros(P.shape[:2])

for i in range(I.shape[2]):
    Px += conv2d(P[:,:,i],w,'same','wrap')**2;
    Py += conv2d(P[:,:,i],w.T,'same','wrap')**2;

Pm = np.sqrt(Px+Py);
Pa = np.arctan2(Py,Px);

plt.imshow(Pm,cmap='gray');
plt.imshow(Pa,cmap='gray');

Qm = to_cart(Pm,I.shape[0],x0,y0,r_min,r_max,bilinear,clip);
Qa = to_cart(Pa,I.shape[0],x0,y0,r_min,r_max,bilinear,clip);

plt.imshow(Qm,cmap='gray');
plt.imshow(Qa,cmap='gray');
