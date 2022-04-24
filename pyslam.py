# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:26:34 2022

@author: James Brodovsky
"""

"""
This module contains several Python based implementations for localization,
mapping, and simultaneous localization and mapping.
"""
import numpy as np

# --- LOCALIZATION METHODS ---------------------------------------------
class KalmanFilter:
    def __init__(self, x_dim=1, u_dim=0, z_dim=1):
        """
        Construct the Kalman Filter. Can either construct by specifying each 
        matrix in the model or by specifying the length of the state vector
        """
        assert x_dim>=1, "State dimension must be an integer greater than or equal to 1."
        assert u_dim>=0, "Control input dimension must be zero or a positive integer."
        assert z_dim>=1, "Observation dimension must be an integer greater than or equal to 1."
        
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        
        self.x = np.zeros(x_dim)
        self.F = np.eye(x_dim)
        if u_dim == 0:
            u_ = 1
        else:
            u_ = u_dim
        self.B = np.zeros((x_dim, u_))
        
        self.P = np.eye(x_dim)
        self.Q = np.zeros_like(self.P)
        
        self.H = np.zeros((z_dim, x_dim))
        self.R = np.zeros(z_dim)
        

    def __repr__(self):
        return f"x: {self.x}\nP:\n{self.P}\nF:\n{self.F}\nH:\n{self.H}\nQ:\n{self.Q}\nR:\n{self.R}\nB:\n{self.B}"
    
    def __str__(self):
        out = f"X:\t{self.x}\nP:\t"
        for i, row in enumerate(self.P):
            if i>0:
                out+=f"\t{row}\n"
            else:
                out+=f"{row}\n"
        return out
    
    def step(self, u=None, z=None):
        if u is None:
            u = np.zeros((1,))
            #(f"U : {u}")
        else:
            u = np.array(u)
            assert u.shape == (self.u_dim,), "Control input dimension mismatch"
        if z is None:
            z = np.zeros((self.z_dim, 1))
        else:
            z = np.array(z)
            assert z.shape == (self.z_dim,), "Observation dimension mismatch"
        self.predict(u)
        self.update(z)
    
    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        y_bar = z - self.H @ self.x 
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x += K @ y_bar
        self.P = (np.eye(self.x_dim) - K @ self.H) @ self.P
        self.y = z - self.H @ self.x

class Particle:
    ID = 0
    def __init__(self, state=None, weight=None, Map=None):
        self.state = state
        self.weight = weight
        self.ID = Particle.ID
        self.Map = Map
        Particle.ID += 1
    def __repr__(self):
        return f"PARTICLE #{self.ID}: {self.state} | {self.weight}"
    def __str__(self):
        return f"{self.state} | {self.weight}"
    def __gt__(self, o):
        return self.weight > o.weight
    def __ge__(self, o):
        return self.weight >= o.weight
    def __eq__(self, o):
        return (self.state == o.state) and (self.weight == o.weight)
