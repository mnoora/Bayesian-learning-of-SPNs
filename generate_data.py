
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Gamma, Gaussian, Poisson, Exponential, Bernoulli, Geometric

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up


from spn.io.Graphics import plot_spn
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
import numpy as np
import matplotlib.pyplot as plt

def generate_data(data_size):

    p1 = Product(children=[Gaussian(scope=1, mean=19, stdev = 2), Poisson(scope=2, mean=1)])
    p2  = Product(children=[Gaussian(scope=1, mean=19, stdev = 2), Poisson(scope=2, mean=1)])
    spn = Sum(weights=[0.2, 0.8], children=[p1, p2])

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)
    
    array = sample_instances(spn, np.array([np.nan, np.nan, np.nan] * data_size).reshape(-1, 3), RandomState(123))

    a = array[:,1:]
    return a


def generate_data_from_example_SPN(data_size):
    p1_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Poisson(scope=1, mean=3)])
    p2_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Poisson(scope=1, mean=3)])

    p3_third_layer = Product(children=[Exponential(scope=2, l=1), Gaussian(scope=3, mean=12, stdev = 1)])
    p4_third_layer = Product(children=[Exponential(scope=2, l=1), Gaussian(scope=3, mean=12, stdev = 1)])
    
    p5_third_layer = Product(children=[Poisson(scope=1, mean=3), Exponential(scope=2, l=1)])
    p6_third_layer = Product(children=[Poisson(scope=1, mean=3), Exponential(scope=2, l=1)])

    p7_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Exponential(scope=3, l=2)])
    p8_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Exponential(scope=3, l=2)])

    s1_second_layer = Sum(weights=[0.5, 0.5], children=[p1_third_layer, p2_third_layer])
    s2_second_layer = Sum(weights=[0.8, 0.2], children=[p3_third_layer, p4_third_layer])
    s3_second_layer = Sum(weights=[0.4, 0.6], children=[p5_third_layer, p6_third_layer])
    s4_second_layer = Sum(weights=[0.5, 0.5], children=[p7_third_layer, p8_third_layer])

    p1_first_layer = Product(children=[s1_second_layer, s2_second_layer])
    p2_first_layer = Product(children=[s3_second_layer, s4_second_layer])
    spn = Sum(weights=[0.4, 0.6], children=[p1_first_layer, p2_first_layer])


    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    array = sample_instances(spn, np.array([np.nan, np.nan, np.nan, np.nan, np.nan] * data_size).reshape(-1, 5), RandomState(123))
    array = array[:,:-1]
    return array

def create_generating_spn():
    p1_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Poisson(scope=1, mean=3)])
    p2_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Poisson(scope=1, mean=3)])

    p3_third_layer = Product(children=[Exponential(scope=2, l=1), Gaussian(scope=3, mean=12, stdev = 1)])
    p4_third_layer = Product(children=[Exponential(scope=2, l=1), Gaussian(scope=3, mean=12, stdev = 1)])
    
    p5_third_layer = Product(children=[Poisson(scope=1, mean=3), Exponential(scope=2, l=1)])
    p6_third_layer = Product(children=[Poisson(scope=1, mean=3), Exponential(scope=2, l=1)])

    p7_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Exponential(scope=3, l=2)])
    p8_third_layer = Product(children=[Gaussian(scope=0, mean=19, stdev = 2), Exponential(scope=3, l=2)])

    s1_second_layer = Sum(weights=[0.5, 0.5], children=[p1_third_layer, p2_third_layer])
    s2_second_layer = Sum(weights=[0.8, 0.2], children=[p3_third_layer, p4_third_layer])
    s3_second_layer = Sum(weights=[0.4, 0.6], children=[p5_third_layer, p6_third_layer])
    s4_second_layer = Sum(weights=[0.5, 0.5], children=[p7_third_layer, p8_third_layer])

    p1_first_layer = Product(children=[s1_second_layer, s2_second_layer])
    p2_first_layer = Product(children=[s3_second_layer, s4_second_layer])
    spn = Sum(weights=[0.4, 0.6], children=[p1_first_layer, p2_first_layer])


    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    return spn