
from webbrowser import get
from node import Leaf, Product, Sum
from spn_leaves import get_all_leaves
from scipy.stats import norm, poisson, gamma, bernoulli, geom, expon
import numpy as np
import copy


def sample_scopes(node, dim):
    for child in node.children:
        if (isinstance(child, Sum)):
            scope_for_region_node_child(node, child, dim)
        elif(isinstance(child, Product)):
            scope_for_partition_node_child(node, child, dim)

#Sum
def scope_for_region_node_child(parent, child_sum, dim):
    sample_scopes(child_sum, dim)

#Product
def scope_for_partition_node_child(parent, partition, dim):
    beta = 1
    partition.new_scope = copy.deepcopy(parent.new_scope)

    if (dim not in partition.new_scope):
        return

    component_counts = {}
    for child in partition.children:
        # Form component counts
        count = len(list(set(partition.scope).intersection(child.scope)))
        if(dim in child.scope):
            count -= 1
        component_counts[child] = count

    sum_of_component_counts = 0

    for child in partition.children:
        sum_of_component_counts += (beta + component_counts[child])

    first_term_cond_prob = {}
    for child in partition.children:
        first_term_cond_prob[child] = (beta + component_counts[child]) / sum_of_component_counts

    second_term_cond_prob = {}
    for child in partition.children:
        f = 1
        if isinstance(child, Leaf):
            data = child.data[dim]
            parameters = child.parameters[dim]

            for key, values in data.items():
                for item in values: 
                    prob_density = PDF_for_likelihood_model(key, item, parameters)
                    if (prob_density != 0):
                        f *= PDF_for_likelihood_model(key, item, parameters)

            second_term_cond_prob[child] = f

        elif isinstance(child, Sum):
            leaves_per_child = []
            get_all_leaves(child, leaves_per_child)

            for leaf in leaves_per_child:
                data = leaf.data[dim]
                parameters = leaf.parameters[dim]

                for key, values in data.items():
                    for item in values:
                        prob_density = PDF_for_likelihood_model(key, item, parameters)
                        if (prob_density != 0):
                            f *= PDF_for_likelihood_model(key, item, parameters)
                
            second_term_cond_prob[child] = f

    conditional_prob = {}
    for child in partition.children:
        conditional_prob[child] = first_term_cond_prob[child]*second_term_cond_prob[child]

    children =  list(conditional_prob.keys())

    
    if (np.sum(list(conditional_prob.values())) != 0):
        # Normalize probabilities
        conditional_children_probabilities = list(conditional_prob.values()) / np.sum(list(conditional_prob.values()))
        chosen = np.random.choice(children, p=conditional_children_probabilities)
    else:
        chosen = np.random.choice(children)
    chosen.new_scope.append(dim)

    sample_scopes(partition, dim)
    

def PDF_for_likelihood_model(model, data_point, parameters):
    p = parameters[model]

    if (model == "Gaussian"):
        return norm.pdf(data_point, loc=p["mean"], scale=p["stdev"])
    elif (model == "Poisson"):
        return poisson.pmf(data_point, p["mean"])
    elif (model == "Gamma"):
        return gamma.pdf(data_point, a=p["a"], scale=1/p["b"])
    elif (model == "Bernoulli"):
        return bernoulli.pmf(data_point, p=p["p"])
    elif (model == "Geometric"):
        return geom.pmf(data_point, p=p["p"])
    elif (model == "Exponential"):
        return expon.pdf(data_point, scale=1/p["l"])
