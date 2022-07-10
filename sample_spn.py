import node
import numpy as np
from numpy.random import uniform,randint
from scipy.stats import gamma, norm, invgamma
from priors import NormalInverseGamma, Gamma, Beta

def create_small_spn():
    leaf_1 = node.Leaf(dims=2,scope=[0])
    leaf_2 = node.Leaf(dims=2,scope=[1])
    leaf_3 = node.Leaf(dims=2,scope=[0])
    leaf_4 = node.Leaf(dims=2,scope=[1])

    product_1 = node.Product(children=[leaf_1, leaf_2],scope=[0,1])
    product_2 = node.Product(children=[leaf_3, leaf_4],scope=[0,1])

    root = node.Sum(children=[product_1,product_2],weights=[0.5, 0.5],scope=[0,1])

    return root


def create_big_SPN():
    leaf_1 = node.Leaf(dims=4,scope=[0])
    leaf_2 = node.Leaf(dims=4,scope=[1])
    leaf_3 = node.Leaf(dims=4,scope=[0])
    leaf_4 = node.Leaf(dims=4,scope=[1])

    leaf_5 = node.Leaf(dims=4,scope=[2])
    leaf_6 = node.Leaf(dims=4,scope=[3])
    leaf_7 = node.Leaf(dims=4,scope=[2])
    leaf_8 = node.Leaf(dims=4,scope=[3])

    leaf_9 = node.Leaf(dims=4,scope=[2])
    leaf_10 = node.Leaf(dims=4,scope=[1])
    leaf_11 = node.Leaf(dims=4,scope=[2])
    leaf_12 = node.Leaf(dims=4,scope=[1])

    leaf_13 = node.Leaf(dims=4,scope=[0])
    leaf_14 = node.Leaf(dims=4,scope=[3])
    leaf_15 = node.Leaf(dims=4,scope=[0])
    leaf_16 = node.Leaf(dims=4,scope=[3])

    product_1 = node.Product(children=[leaf_1, leaf_2],scope=[0,1])
    product_2 = node.Product(children=[leaf_3, leaf_4],scope=[0,1])
    product_3 = node.Product(children=[leaf_5, leaf_6],scope=[2,3])
    product_4 = node.Product(children=[leaf_7, leaf_8],scope=[2,3])

    product_5 = node.Product(children=[leaf_9, leaf_10],scope=[1,2])
    product_6 = node.Product(children=[leaf_11, leaf_12],scope=[1,2])
    product_7 = node.Product(children=[leaf_13, leaf_14],scope=[0,3])
    product_8 = node.Product(children=[leaf_15, leaf_16],scope=[0,3])

    sum_1 = node.Sum(children=[product_1,product_2],weights=[0.5, 0.5],scope=[0,1])
    sum_2 = node.Sum(children=[product_3,product_4],weights=[0.5, 0.5],scope=[2,3])

    sum_3 = node.Sum(children=[product_5,product_6],weights=[0.5, 0.5],scope=[1,2])
    sum_4 = node.Sum(children=[product_7,product_8],weights=[0.5, 0.5],scope=[0,3])

    product_1_first_layer = node.Product(children=[sum_1, sum_2],scope=[0,1,2,3])
    product_2_first_layer = node.Product(children=[sum_3, sum_4],scope=[0,1,2,3])

    root = node.Sum(children=[product_1_first_layer,product_2_first_layer],weights=[0.5, 0.5],scope=[0,1,2,3], new_scope=[0,1,2,3])

    return root