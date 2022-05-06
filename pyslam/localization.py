
"""
This module contains several Python based implementations for localization. Including a basic Kalman Filter, and base classs for an Extended Kalman Filter and a Particle Filter. 

Planned future work includes:
    * completion of the base particle filter
    * A base Unscented Kalman Filter
    * Implementation of a geo-physical particle filter for topographic, magnetic, and gravimetric navigation
"""
import numpy as np

# --- KALMAN FILTER ----------------------------------------------------------
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
    
class EKFTwoD(ExtendedKalmanFilter):
    '''
    Simple Cartesian plane EKF: x and y position, body frame velocity, heading,
    angular velocity. Measures distance and bearing to a fixed landmark. Here 
    as an example
    '''
    
    # ==== WORK IN PROGRESS ===================================================
    def f(self, u):
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
        self.H = np.array(H)
# --- PARTICLE FILTERS --------------------------------------------------------
class Particle:
    """
    A basic particle object for use in a particle filter.
    """
    
    ID = 0
    def __init__(self, state=None, weight=None, Map=None):
        '''
        Construct a particle.

        :param state: the state vector
        :type state: array or array-like
        :param weight: particle's weight or importance factor
        :type weight: float
        :param Map: the particle's map version (used for SLAM)
        '''
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
        '''
        Compares particles based on weight.
        '''
        return self.weight > o.weight
    def __ge__(self, o):
        '''
        Compares particles based on weight.
        '''
        return self.weight >= o.weight
    def __eq__(self, o):
        '''
        Compares particles based on state and weight. Will return true if 
        two particles have the exact same state vector and weight.
        '''
        return (self.state == o.state) and (self.weight == o.weight)

class ParticleFilter:
    '''
    This is a base class particle filter that contains many of the re-usable 
    elements common across implementaitons of particle fitlers. 
    '''
    def __init__(self, n_particles=100, resampling_type='Random', 
                 state_estimation_method='WeightedAverage', 
                 resampling_threshold=0):
        '''
        Constructor.
        '''
        assert n_particles>1, 'Number of particles must be greater than one.'
        self.n_particles = n_particles
        self.resampling_type = resampling_type
        self.state_estimation_method = state_estimation_method
        self.state_estimate = None
        self.resampling_threshold = resampling_threshold
        #self.resampling_noise = resampling_noise
        self.particles = []
        for i in range(self.n_particles):
            self.particles.append(Particle(state=None, weight=1/self.n_particles, Map=None))
        
    def initialize(self, method='about', resampling_noise=None, **kwargs ):
        '''
        Used to initialize the particle field

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'about'.
        resampling_noise : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if method=='about' or method=='About':
            mu, sigma = kwargs.items()
            for particle in self.particles:
                particle.state = np.random.normal(mu, sigma)
        elif method=='fixed' or method=='Fixed':
            pass
        # add in addtional initialization methods if desired. 
        self.estimate_states()
        
    def step(self, Input=None, Observations=None):
        # Propagate
        for particle in self.particles:
            self.propagate(particle, Input)
        # Measurement update
        if Observations is not None:
            self.update(Observations)
        # Resample
        Neff = 1 / self.get_sum_weight_squared()
        Nthresh = self.resampling_threshold * self.n_particles # to resample every step set resampling_threshold to 1
        if Neff < Nthresh:
            self.resample()
        # Update state estimate
        self.estimate_states()
        
    def propagate(self, particle, Input=None):
        print("Base particle filter class does not have a propagation model.")
        
    def update(self, observations):
        for particle in self.particles:
           self.update_weight(particle, observations)
        if self.get_total_weight() <= 0:
            raise Exception()
            
    def update_weight(self, particle, observations):
        print("Base particle filter does not have an observation model.")
        
    def estimate_states(self):
        #estimate = np.zeros_like(self.particles[0].state)
        if self.state_estimation_method == 'HighestWeight':
            self.state_estimate = self.get_highest_weighted_particle()
        elif self.state_estimation_method == 'Average':
            average = np.zeros_like(self.particles[0].state)
            for particle in self.particles:
                average += particle.state()
            average /= self.n_particles
            self.state_estimate = average
        elif self.state_estimate_method == 'WeightedAverage':
            average = np.zeros_like(self.particles[0].state)
            for particle in self.particles:
                average += particle.state * particle.weight
            self.state_estimate = average
    
    def get_sum_weight_squared(self):
        w = 0.0
        for particle in self.particles:
            w += particle.weight**2
        return w
    def get_total_weight(self):
        w = 0.0
        for particle in self.particles:
            w += particle.weight
        return w
    def get_highest_weighted_particle(self):
        particles = sorted(self.particles)
        return particles[-1]
    
        