"""
Examples for how to implement and use the localization tools. In this module various implementations of a 
"""
import numpy as np
from gaussian import ExtendedKalmanFilter, UnscentedKalmanFilter
from nonparameteric import ParticleFilter

class EKF(ExtendedKalmanFilter):
    '''
    Simple Cartesian plane EKF: x and y position, body frame velocity, heading,
    angular velocity. Measures distance and bearing to a fixed landmark. Here 
    as an example
    '''
    # ==== WORK IN PROGRESS ===================================================
    '''def f(self, u):
        dt = u[-1]
        x = self.x[0]
        y = self.y[1]
        v = self.x[2]
        q = self.x[3]
        w = self.x[4]
        x += ((v - y*w)*np.cos(q) - x*w*np.sin(q))*dt
        y += ((v - y*w)*np.sin(q) - x*w*np.cos(q))*dt
        v += 0
        q += w*dt
        w += 0
        return np.array([x, y, v, q, w])
    def h(self, x):
        return np.array([0, 0])
    def calculate_F(self, u):
        pass
    def calculate_H(self, z):
        r = z[0]
        H = [[np.cos(z[1]), np.sin(z[1]), 0, 0, 0],
             [-np.sin(z[1]) / r, np.cos(z[1]) / r, 0, 0, 0]]
        self.H = np.array(H)'''

class UKF(UnscentedKalmanFilter):
    pass

class TwoDPF(ParticleFilter):
    pass

if __name__ == '__main__':
    print('Hello world!')