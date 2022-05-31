"""
Gaussian module contains bases for Gaussian implementations of Bayes Filter.
Currently, a basic Kalman Filter and an EKF base are implemented. Future
development plans include a UKF and Information Filter
"""
import numpy as np

class KalmanFilter:
    """
    A basic Kalman Filter implemented using NumPy. This class is used to ensure
    that the matricies are properly constructed with appropriately sized 
    dimensions and cycle through the Kalman Filter. Currently also exploring
    if it can be used as a base class for the Kalman Filter derivatives (EKF and
    UKF) however those also require customization that is unique to their
    implementation.
    """
    def __init__(self, x_dim=1, u_dim=0, z_dim=1):
        '''
        Construct the Kalman filter by providing dimensions.

        :param x_dim: Length of the state vector. The default is 1. Used to create
        :type x_dim: INT
        :param u_dim: Length of the control input vector. The default is 0.
        :type u_dim: INT
        :param z_dim: Length of the measurement or observation vector. The default is 1.
        :type z_dim: INT
        
        :returns: The Kalman Filter object
        
        '''
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
        print("Kalman Filter constructed. Please manaully initialize matrix values.")
        

    def __repr__(self):
        return f"x: state estimate\n{self.x}\nP: covariance\n{self.P}\nF: state transition matrix\n{self.F}\nH: observation model matrix\n{self.H}\nQ: process noise covariance\n{self.Q}\nR: observation noise covariance\n{self.R}\nB: control input model\n{self.B}"
    
    def __str__(self):
        out = f"X:\t{self.x}\nP:\t"
        for i, row in enumerate(self.P):
            if i>0:
                out+=f"\t{row}\n"
            else:
                out+=f"{row}\n"
        return out
    
    def step(self, u=None, z=None):
        """
        Executes one cycle of the Kalman Filter

        Parameters
        ----------
        :param u: the control input vector
        :type u: array or array-like
        :param z: the measurement or observation vector
        :type z: array or array-like
        """
        
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
        '''
        Predict (propagate) the Kalman Filter's state estimate. This method
        should not be called directly. Call `step` instead.
        
        :param u: the control input vector
        :type u: array or array-like
        '''
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        '''
        The measurement update function. This method should not be called 
        directly. Call `step` instead.

        :param z: the measurement or observation vector
        :type z: array or array-like
        '''
        y_bar = z - self.H @ self.x 
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x += K @ y_bar
        self.P = (np.eye(self.x_dim) - K @ self.H) @ self.P
        self.y = z - self.H @ self.x

class ExtendedKalmanFilter(KalmanFilter):
    # Use inherited constructor
    '''
    Base level class. Due to non-linear models, specific implementations must
    be built with unique models. Use this as a base EKF class and extend it by
    inheriting an over riding the f(), h(), and calculate_F(), and 
    calculate_H() functions.
    '''
    
    def predict(self, u):
        # propagate using non-linear function. I use a convetion of including
        # a time duration as the last element of the control input if needed
        self.x = self.f(u)
        # update F matrix
        self.calculate_F(u)
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        # calculate innovation y = z - h()
        y_bar = z - self.h(self.x)
        # update H matrix
        self.calculate_H(z)
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x += K @ y_bar
        self.P = (np.eye(self.x_dim) - K @ self.H) @ self.P

    def f(self, u):
        raise Exception("Must define state transition function")
    def h(self, z):
        raise Exception("Must define observation model")
    def calculate_F(self, u):
        raise Exception("Must define state transition Jacobian")
    def calculate_H(self, z):
        raise Exception("Must define observation Jacobian")
    
class UnscentedKalmanFilter(KalmanFilter):
    '''
    Base level class. Due to non-linear models, specific implementations must
    be built with unique models. Implementation to be added in the future.
    State is store row-wise to leverage NumPy linear algebra operations.
    '''
    def __init__(self, x_dim=1, u_dim=0, z_dim=1, alpha=1, beta=2, kappa=1):
        super().__init__(x_dim, u_dim, z_dim)
        self.Alpha_sqrd = alpha**2
        self.Beta = beta
        self.Kappa = kappa
        self.Lambda = self.Alpha_sqrd*(x_dim + self.Kappa) - self.x_dim
    
    def initialize(self, mu, Sigma):
        mu = np.array(mu)
        Sigma = np.array(Sigma)
        assert len(mu) == self.x_dim
        assert Sigma.shape == (self.x_dim, self.x_dim)
        self.x = mu
        self.P = Sigma
        self.X = np.zeros(((2*self.x_dim + 1), self.x_dim))
        self.w_m = [self.Lambda / (self.x_dim + self.Lambda)]
        self.w_c = self.w_m[0] + 1 - self.Alpha_sqrd + self.Beta
        for i in range(1, self.x_dim*2):
            w = 1 / (2*(self.x_dim + self.Lambda))
            self.w_m.append(w)
            self.w_c.append(w)
        self.w_m = np.array(self.w_m)
        self.w_c = np.array(self.w_c)
        
    def calculate_sigma_points(self):
        self.X[0, :] = self.x
        spread = np.sqrt((self.x_dim + self.Lambda) * self.P)
        self.X[1:(self.x_dim + 1), :] = self.x + spread
        self.X[(self.x_dim + 1):,  :] = self.x - spread
        
    def predict(self, u):
        self.calculate_sigma_points()
        X_bar_star = self.g(u)
        self.x = self.w_m @ X_bar_star
        self.P = self.w_c * ((X_bar_star - self.x) @ (X_bar_star - self.x).T) + self.R
        self.calculate_sigma_points()
        
    def update(self, z):
        Z_bar = self.h()
        z_hat = self.w_m @ Z_bar
        S = self.w_c * ((Z_bar - z_hat) @ (Z_bar - z_hat).T) + self.Q
        P_bar = self.w_c * ((self.X - self.x) @ (Z_bar - z_hat).T)
        K = P_bar @ np.linalg.inv(S)
        self.x += K @ (z - z_hat)
        self.P += -K@S@K.T
        
    def g(self, u):
        return self.X
    def h(self):
        return np.zeros_like(self.X)

class InformationFilter():
    '''
    Implementation to be added in the future
    '''
    pass