from email import iterators
from tracemalloc import start
import numpy as np
from numpy.random import randint
import sample_spn
import metropolis_hastings
from collections import defaultdict
from node import Sum, Product, Leaf
from generate_data import generate_data, generate_data_from_example_SPN
from spn_leaves import *
from parameter_updates import *
from structure_learning import *


def start_learning():
    data_size = 7000
    max_iters = 10000

    #spn = sample_spn.create_small_SPN()
    spn = sample_spn.create_big_SPN()

    #data_set = generate_data(data_size)
    data_set = generate_data_from_example_SPN(data_size)

    learn_data(spn, data_set, max_iters)

def clear_leaf_data(spn, dims):
    leafs = []
    get_all_leaves(spn, leafs)
    likelihoods = ["Gaussian", "Exponential", "Poisson", "Gamma", "Bernoulli", "Geometric"]
    for leaf in leafs:
        for d in range(dims):
            for l in likelihoods:
                leaf.data[d][l] = []

def clear_substitute_scopes(spn):
    nodes = []
    get_all_nodes(spn, nodes)
    for node in nodes:
        node.new_scope = []

def assign_substitute_scopes_to_node_scopes(spn):
    nodes = []
    get_all_nodes(spn, nodes)
    for node in nodes:
        node.scope = node.new_scope

        if isinstance(node, Leaf):
            node.scope_store.append(node.scope)



def learn_data(spn, data_set, max_iters):
    z_count = defaultdict(int)
    z_assignments = defaultdict(int)

    dims = len(data_set[0])
    likelihoods = ["Gaussian", "Poisson", "Gamma", "Bernoulli", "Geometric", "Exponential"]
    
    z_count = init_counter_for_z(spn,z_count)
    for i in range(max_iters):
        clear_leaf_data(spn, dims)
        clear_substitute_scopes(spn)
        # Init spn root scope
        spn.new_scope = list(range(dims))

        # PARAMETER LEARNING
        z_count = init_counter_for_z(spn,z_count)
        
        if i % 1000 == 0: print(50*"-")
        if i % 1000 == 0: print("SAMPLING z and s, iteration:", i)
        for row in data_set:
            z_count[spn] += 1
            ancestral_sampling("z",z_count,z_assignments,spn,row, likelihoods)
        
        sample_z_for_empty(z_count,z_assignments,spn)

        set_leaf_data_size(spn, dims)
        leaves = []
        get_all_leaves(spn, leaves)
        if i % 1000 == 0: print("SAMPLING HYPERPARAMETERS, iteration:", i)
        if i % 1000 == 0: print_spn(spn)
        for d in range(dims):
            for leaf in leaves:
                sample_likelihood_weights(leaf, d)
                update_hyperparameters(leaf, d)

        for leaf in leaves:
            leaf.all_likelihood_weights.append(leaf.likelihood_weights)

        if i % 1000 == 0: print("SAMPLING WEIGHTS, iteration:", i)
        sample_weights(spn, z_count)

        if i % 1000 == 0: metropolis_hastings.get_acceptance_ratio()
        
        # STRUCTURE LEARNING
        for dim in range(dims):
            sample_scopes(spn, dim)
        
        assign_substitute_scopes_to_node_scopes(spn)

        if i % 500 == 0:
            save_sum_weights(spn)
            save_leaf_parameters(spn, dims)
            save_leaf_scopes(spn)
            save_sum_mh_prob(spn)
            save_likelihood_weights(spn)

    
    print(50*"-")
    save_sum_weights(spn)
    save_leaf_parameters(spn, dims)
    save_leaf_scopes(spn)
    save_sum_mh_prob(spn)
    save_likelihood_weights(spn)

    metropolis_hastings.get_acceptance_ratio()


def ancestral_sampling(z,z_count,z_assignments,node,row, likelihoods):
    root = node

    if(isinstance(root,Product)):
        for child in root.children:
            z_count[child] += 1
            ancestral_sampling("z",z_count,z_assignments,child,row, likelihoods)

    if(isinstance(root,Sum)):
        z = metropolis_hastings.metropolis_hastings("z",z_assignments[root],weights=root.weights, node=root)
        z_count[root.children[z]] += 1
        z_assignments[root] = z
        ancestral_sampling("z",z_count,z_assignments,root.children[z],row, likelihoods)

    if(isinstance(root,Leaf)):
        for dim, point in enumerate(row, start=0):
            s = metropolis_hastings.metropolis_hastings("s", root.last_likelihood[dim], likelihood_weights=root.likelihood_weights[dim], node=root)
            root.last_likelihood[dim] = s
            likelihood = likelihoods[s]

            root.data[dim][likelihood].append(point)

def sample_z_for_empty(z_count,z_assignments,spn):
    for key in z_count:
        if (z_count[key] == 0 and isinstance(key,Sum)):
            z = metropolis_hastings.metropolis_hastings("z",z_assignments[key],weights=key.weights)
            z_count[key] += 1
            z_assignments[key] = z

def sample_weights(node, z_count):
    if isinstance(node, Leaf):
        return
    alpha = 1

    count = []

    for child in node.children:
        count.append(z_count[child])

    alphas_with_count = [x+alpha for x in count]

    if (isinstance(node,Sum)):
        weights = metropolis_hastings.metropolis_hastings("w",node.weights, alphas=alphas_with_count, node=node)
        node.weights = weights
        node.markov_chain.append(weights)
        for child in node.children:
            sample_weights(child, z_count)

    if (isinstance(node,Product)):
        for child in node.children:
            sample_weights(child, z_count)

def sample_likelihood_weights(leaf, dim):
    alpha = 1
    likelihoods = leaf.likelihoods
    parameters = []
    for likelihood in likelihoods:
        data = leaf.data[dim][likelihood]
        parameters.append(len(data) + alpha)
    likelihood_weights = metropolis_hastings.metropolis_hastings("l", leaf.likelihood_weights[dim], likelihood_count=parameters)
    leaf.likelihood_weights[dim] = likelihood_weights

def init_counter_for_z(root,z_count):

    if (isinstance(root, Sum) or isinstance(root, Product)):
        z_count[root] = 0
        for child in root.children:
            init_counter_for_z(child, z_count)
    elif (isinstance(root, Leaf)):
        z_count[root] = 0

    return z_count

start_learning()
