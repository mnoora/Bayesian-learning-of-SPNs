from numpy.random import multinomial, dirichlet
import numpy as np
import random
import copy


class Node(object):
    id_counter = 0
    def __init__(self, scope=None, new_scope=None, parent= None):
        self.id = Node.id_counter
        Node.id_counter += 1

        if scope is None:
            scope = []
        self.scope = scope
        self.new_scope=new_scope
        self.parent = parent

    @property
    def name(self):
        return '{}_node_{}'.format(self.__class__.__name__, self.id)

    def __repr__(self):
        return self.name

class Product(Node):
    def __init__(self, children=None, prior=None, dims=1,scope=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children

        self.prior = prior

        if scope is None:
            scope = list(range(dims))
        self.scope = scope
        self.mcmcmc_chain = {}

class Sum(Node):
    def __init__(self,children=None, weights=None, markov_chain=None, dims = 1, scope=None, new_scope=None, mh_prob_z=None, mh_prob_w = None, mh_prob_w_all = None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children

        if weights is None:
            weights = []
        self.weights = weights

        if markov_chain is None:
            markov_chain = []
        self.markov_chain = markov_chain

        self.scope = scope
        self.new_scope=new_scope

        if new_scope is None:
            new_scope = []

        if mh_prob_z is None:
            mh_prob_z = []

        if mh_prob_w is None:
            mh_prob_w = []

        if mh_prob_w_all is None:
            mh_prob_w_all =[]
        
        self.mh_prob_w = mh_prob_w
        self.mh_prob_z = mh_prob_z
        self.mh_prob_w_all = mh_prob_w_all

class Leaf(Node):
    def __init__(self, parameters=None, prior=None, data=None, likelihood=None, hyperparameters=None, markov_chain=None, dims=None, scope=None, likelihood_weights=None):
        Node.__init__(self)
    
        if (dims == None):
            self.dims = 1

        self.dims = dims    
        
        p = {"Gaussian":    {"mean": 1,
                             "stdev":1},
             "Poisson":     {"mean": 1},
             "Gamma":       {"a": 1,
                             "b": 1},
             "Bernoulli":   {"p": 1},
             "Geometric":   {"p": 1},
             "Exponential": {"l": 1}}
        if parameters is None:
            self.parameters = {}

            for d in range(dims):
                self.parameters[d] = copy.deepcopy(p)
        else:
            self.parameters = parameters

        h = {"Gaussian":    {"mu":1,
                             "v":1,
                             "a":1,
                             "b":1},
             "Poisson":     {"a":2,
                             "b":1},
             "Gamma":       {"a":1,
                             "b":1},
             "Bernoulli":   {"a":1,
                             "b":1},
             "Geometric":   {"a":1,
                             "b":1},
             "Exponential": {"a":1,
                             "b":1}}

        self.hyperparameters = {}
        for d in range(dims):
            self.hyperparameters[d] = copy.deepcopy(h)
        
        pr = {"Normal-Inverse-Gamma": {"mu":1,
                                       "v":1,
                                       "a":1,
                                       "b":1},
              "Gamma":                {"a":1,
                                       "b":1},
              "Normal":               {"sd":1,
                                       "var":1},
              "Beta":                 {"a":1,
                                       "b":1}}

        self.prior = {}
        for d in range(dims):
            self.prior[d] = copy.deepcopy(pr)

        self.likelihoods = ["Gaussian", "Poisson", "Gamma", "Bernoulli", "Geometric", "Exponential"]
        
        if likelihood_weights is None:
            self.likelihood_weights = {}
            for d in range(dims):
                self.likelihood_weights[d] = []
        
            for key in self.likelihood_weights:
                self.likelihood_weights[key] = dirichlet(np.ones(6),size=1)[0]
            
        else:
            self.likelihood_weights = likelihood_weights
        
        self.last_likelihood = {}
        for d in range(dims):
            self.last_likelihood[d] = random.randint(0,5)
    
        self.all_likelihood_weights = []

        self.data = {}

        da = {"Gaussian": [],
             "Poisson":   [],
             "Gamma":     [],
             "Bernoulli":   [],
             "Geometric":   [],
             "Exponential": []}

        for d in range(dims):
            self.data[d] = copy.deepcopy(da)

        self.data_size = {}

        for d in range(dims):
            self.data_size[d] = copy.deepcopy(da)

        self.markov_chain = {}
        for d in range(dims):
            self.markov_chain[d] = copy.deepcopy(da)

        if scope is None:
            scope = list(range(dims-1))
        self.scope = scope

        self.new_scope = []

        self.scope_store = []