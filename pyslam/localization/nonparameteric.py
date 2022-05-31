"""
Module for nonparameteric implementations of Bayesian Filters. Specifically, a
particle filter base and histogram filter are contained in this module.
"""
import numpy as np

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
        self.state = np.array(state, dtype='float64')
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
    elements common across implementaitons of particle fitlers. It is designed
    to function as a pseudo-abstract class. Two methods propagate() and
    update_weights() will always be implementation-sepcific, as well as the 
    given state space, control input, and measurements. Thus this class is to 
    be used as an abstract base that takes care of the backend for your specific
    implementation. For your implementation, inherit from this class, then override
    the two abstract methods.
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
        #for i in range(self.n_particles):
        #    self.particles.append(Particle(state=None, weight=1/self.n_particles, Map=None))
        
    def initialize(self, method='about', resampling_noise=0, **kwargs ):
        '''
        Used to initialize the particle field.

        :param method: initializaiton method description; currently available are: "about", "fixed", and "uniform".
        :type method: string
        :param resampling_noise: noise or "jitter" added to the particle state on resampling.
        :type resampling_noise: float or array of floats

        '''
        self.particles = [] # clear particle list to prevent duplicate initializations
        if method=='about' or method=='About':
            mu, sigma = kwargs.items()
            for i in range(self.n_particles):
                self.particles.append(Particle(np.random.normal(mu, sigma), 1.0/self.n_particles))
        elif method=='fixed' or method=='Fixed':
            particles = kwargs['particles']
            self.n_particles = len(particles)
            for p in particles:
                P = Particle(p, 1.0/self.n_particles)
                self.particles.append(P)
        elif method=='across' or method=='Across':
            limits = kwargs['limits']
            N = kwargs['N']
            is_uniform = kwargs['is_uniform']
            states = []
            self.n_particles = N
            for limit in limits:
                assert len(limit) == 2
                state = []
                if is_uniform:
                    state = np.linspace(limit[0], limit[1], self.n_particles)
                else:
                    state = np.random.uniform(limit[0], limit[1], self.n_particles)
                states.append(state)
            states = np.stack(states)
            self.initialize(method='fixed', particles=states.T)
        
        self.resampling_noise = np.array(resampling_noise)
        self.estimate_states()
        
    def step(self, Input=None, Observations=None):
        '''
        Used to run a single cycle of the particle filter algorithm
        
        :param Input: control input vector
        :type Input: array-like, converted to numpy array in method
        :param Observation: observation or measurment vector
        :type Observation: array-like, converted to numpy array in method
        '''
        # Propagate
        for particle in self.particles:
            self.propagate(particle, np.array(Input))
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
        '''
        Private abstract method used to propagate the particle field. Should be overridden in specific sub-class implementation.
        '''
        
        print("Base particle filter class does not have a propagation model.")
        raise Exception()
        
    def update(self, observations):
        '''
        Private method used to update the weights of the particles based on 
        measurements recieved.
        '''
        for particle in self.particles:
           self.update_weight(particle, observations)
        if self.get_total_weight() <= 0:
            raise Exception()
        self.noramlize_weights()
            
    def update_weight(self, particle, observations):
        '''
        Private abstract method for the measurement or observation model.
        '''
        print("Base particle filter does not have an observation model.")
        raise Exception()
        
    def estimate_states(self):
        '''
        Method used to determine the navigation fix or solution of the particle
        filter based on the estimation method specified.
        '''
        #estimate = np.zeros_like(self.particles[0].state)
        if self.state_estimation_method == 'HighestWeight':
            self.state_estimate = self.get_highest_weighted_particle()
        elif self.state_estimation_method == 'Average':
            average = np.zeros_like(self.particles[0].state)
            for particle in self.particles:
                average += particle.state()
            average /= self.n_particles
            self.state_estimate = average
        elif self.state_estimation_method == 'WeightedAverage':
            average = np.zeros_like(self.particles[0].state)
            for particle in self.particles:
                average += particle.state * particle.weight
            self.state_estimate = average
    
    def resample(self):
        '''
        This method calls the specified resampling method of the particle filter.
        '''
        if self.resampling_type == 'Residual':
            self.residual_resample()
        elif self.resampling_type == 'Multinomial':
            self.multinomial_resample()
        else:
            self.random_resample()
    
    def random_resample(self):
        '''
        Resample particles randomly with replacement using NumPy
        '''
        self.particles.sort(reverse=True)
        new_particles = []
        for i in range(0, self.n_particles):
            r = np.random.random()
            P = self.particles[next(x[0] for x in enumerate(self.particles) if x[1] > r)]
            P.state = np.normal(P.state, self.resampling_noise)
            new_particles.append(P)
        self.particles = new_particles
    
    def multinomial_resample(self):
        '''
        Resample particles using multinomial method
        '''
        new_particles = []
        index = int(np.random.random() * self.n_particles)
        highest_particle = self.get_highest_weighted_particle()
        max_weight = highest_particle.weight
        for i in range(0, self.n_particles):
            beta = np.random.random() * 2.0 * max_weight
            while beta > self.particles[index].weight:
                beta -= self.particles[index].weight
                index = (index +1) % self.n_particles
            new_state = np.random.normal(self.particles[index].state, self.resampling_noise)
            new_particles.append(Particle(new_state, self.particles[index].weight))
        self.particles = new_particles
        
    def residual_resample(self):
        '''
        Resample particles using residual method
        '''
        new_particles = []
        # Deterministic
        weights = [particle.weight for particle in self.particles]
        duplicaitons = [int(dups) for dups in [w*self.n_particles for w in weights]] # number of deterministic duplicates
        indecies = [i for i, e in enumerate(duplicaitons) if e >= 1]                 # indecies of deterministic duplicates
        for i in range(len(indecies)):
            for dups in indecies[i]:
                new_state = np.random.normal(self.particles[indecies[i]].state, self.resampling_noise)
                new_particles.append(Particle(new_state, weights[indecies[i]]))
        # Stochastic
        m = self.n_particles - len(indecies) # number of remaining particles to resample
        weights_modified = self.n_particles * np.array(weights) - np.array(duplicaitons)
        index = int(np.random.random()*self.n_particles)
        max_weight = max(weights_modified)
        for i in range(m):
            beta = np.random.random() * 2.0 * max_weight
            while beta > self.particles[index].weight:
                beta -= self.particles[index].weight
                index = (index +1) % self.n_particles
            new_state = np.random.normal(self.particles[index].state, self.resampling_noise)
            new_particles.append(Particle(new_state, self.particles[index].weight))
        self.particles = new_particles       
        
    def get_sum_weight(self):
        '''
        Returns the sum of the weights of the particle field. When normalized 
        this should be equal to 1.0
        '''
        return sum([particle.weight for particle in self.particles])

    def get_sum_weight_squared(self):
        '''
        Returns the sum of the squares of the particles' weights.

        '''
        return sum([particle.weight**2 for particle in self.particles])
    
    def get_highest_weighted_particle(self):
        '''
        Returns highest weighted particle.
        '''
        particles = sorted(self.particles)
        return particles[-1]
    
    def normalize_weights(self):
        '''
        Normailzes the weights of all particles to [0.0, 1.0)
        '''
        total = self.get_sum_weight()
        for particle in self.particles:
            particle.weight /= total
            
class HistogramFilter():
    '''
    To be implemented in the future
    '''
    pass