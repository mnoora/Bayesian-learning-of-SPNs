import numpy as np
from numpy.random import uniform,randint, choice
from numpy.random import multinomial, dirichlet
import random
from scipy.stats import dirichlet
from node import Sum

random.seed(921)
np.random.seed(921)

# Quantiles are weights and alpha is the alpha + counters
def target_PDF_for_Dirichlet_dist(quantiles, alpha):
    return dirichlet.pdf(quantiles, alpha)


def proposal_dist_for_w(alphas):
    v = np.random.uniform(low=0.3, high=1.2) + randint(0,400)
    list_of_v =[v] * len(alphas)
    return np.random.dirichlet(list_of_v)
    

def proposal_dist_for_l(count):
    length = len(count)
    params = randint(1,20,length)
    sample = [random.gammavariate(a, 1) for a in params]
    sample = [v / sum(sample) for v in sample]
    return sample


# sampling from “discrete uniform” distribution
def proposal_dist_for_z(weights):
    len_weights = len(weights)
    sample = randint(0,len_weights)
    return sample

def accept(acceptance_ratio):
    u = uniform()
    if(u <= acceptance_ratio):
        return True
    else:
        return False


def calculate_acceptance_ratio_for_z(x_0,x_new,target_PDF):

    acceptance_ratio = target_PDF[x_new]/target_PDF[x_0]
    acceptance_ratio = min(1, acceptance_ratio)

    return acceptance_ratio

def calculate_acceptance_ratio_for_w(x_0,x_new, target_PDF, alphas, node):

    acceptance_ratio = (target_PDF(x_new, alphas)/target_PDF(x_0, alphas))
    acceptance_ratio = min(1, acceptance_ratio)

    if isinstance(node, Sum):
        node.mh_prob_w_all.append(target_PDF(x_new, alphas))
    
    if accept(acceptance_ratio) and isinstance(node, Sum):
        node.mh_prob_w.append(target_PDF(x_new, alphas))
        node.mh_prob_z.append(acceptance_ratio)
        
    return acceptance_ratio


def metropolis_hastings(sampling_type,init_state, *args, **kwargs):

    weights = kwargs.get('weights', None)
    alphas = kwargs.get('alphas', None)
    likelihood_weights = kwargs.get("likelihood_weights", None)
    likelihood_count = kwargs.get("likelihood_count", None)
    node = kwargs.get("node", None)

    x_0 = init_state

    if(sampling_type == "z"):
        proposal = proposal_dist_for_z
        target_PDF = weights
        x_new = proposal(weights)
        acceptance_ratio = calculate_acceptance_ratio_for_z(x_0,x_new, target_PDF)

    elif(sampling_type =="w"): 
        proposal = proposal_dist_for_w
        target_PDF = target_PDF_for_Dirichlet_dist
        x_new = proposal(alphas)
        acceptance_ratio = calculate_acceptance_ratio_for_w(x_0, x_new, target_PDF, alphas, node)

    elif(sampling_type == "s"):
        proposal = proposal_dist_for_z
        target_PDF = likelihood_weights
        x_new = proposal(likelihood_weights)
        acceptance_ratio = calculate_acceptance_ratio_for_z(x_0, x_new, target_PDF)

    elif(sampling_type == "l"):
        proposal = proposal_dist_for_l
        target_PDF = target_PDF_for_Dirichlet_dist
        x_new = proposal(likelihood_count)
        acceptance_ratio = calculate_acceptance_ratio_for_w(x_0,x_new, target_PDF, likelihood_count, node)

    accepted = accept(acceptance_ratio)

    set_acceptance_ratio(accepted, sampling_type)
    
    if(accepted):
        return x_new
    else:
        return x_0

accepted_z = 0
accepted_w = 0
accepted_s = 0
accepted_l = 0

all_z = 0
all_w = 0
all_s = 0
all_l = 0

def set_acceptance_ratio(accepted, sampling_type):
    global accepted_w, accepted_z, accepted_s, accepted_l,  all_w, all_z, all_s, all_l

    if(sampling_type == "w"):
        all_w += 1
        if(accepted):
            accepted_w += 1
    if(sampling_type == "z"):
        all_z += 1
        if(accepted):
            accepted_z += 1
    if(sampling_type == "s"):
        all_s += 1
        if(accepted):
            accepted_s += 1
    if(sampling_type == "l"):
        all_l += 1
        if(accepted):
            accepted_l += 1

def get_acceptance_ratio():
    print("For z:")
    print(accepted_z/all_z)
    print("For w:")
    print(accepted_w/all_w)
    print("For s:")
    print(accepted_s/all_s)
    print("For l:")
    print(accepted_l/all_l)


def gibbs_sampling(sampling_type,init_state, *args, **kwargs):

    weights = kwargs.get('weights', None)
    alphas = kwargs.get('alphas', None)
    likelihood_weights = kwargs.get("likelihood_weights", None)
    likelihood_count = kwargs.get("likelihood_count", None)
    node = kwargs.get("node", None)

    x_0 = init_state

    if(sampling_type == "z"):
        x_new = choice(range(len(weights)), 1, p=weights)

    elif(sampling_type =="w"): 
        x_new = np.random.dirichlet(alphas)

    elif(sampling_type == "s"):
        x_new = choice(range(len(likelihood_weights)), 1, p=likelihood_weights)

    elif(sampling_type == "l"):
        x_new = np.random.dirichlet(likelihood_count)

    return x_new