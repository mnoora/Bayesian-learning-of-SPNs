import matplotlib.pyplot as plt
from node import Leaf, Product, Sum
import numpy as np

def get_all_leaves(root, list_of_leaves):
    if(isinstance(root, Leaf)):
        list_of_leaves.append(root)

    if(isinstance(root,Product) or isinstance(root, Sum)):
        for child in root.children:
            get_all_leaves(child, list_of_leaves)

def get_all_sums(root, list_of_sums):
    if(isinstance(root, Sum)):
        list_of_sums.append(root)
        for child in root.children:
            get_all_sums(child, list_of_sums)

    if(isinstance(root,Product)):
        for child in root.children:
            get_all_sums(child, list_of_sums)

def get_all_nodes(root, list_of_nodes):
    list_of_nodes.append(root)
    if(isinstance(root, Sum) or isinstance(root, Product)):
        for child in root.children:
            get_all_nodes(child, list_of_nodes)

def print_spn(root):
    
    if(isinstance(root, Leaf)):
        print("LEAF: ", root, "with parameters:", root.parameters)
    elif(isinstance(root, Sum)):
        print("SUM: ", root, "with weights:", root.weights)
    elif(isinstance(root, Product)):
        print("PRODUCT: ", root)

    if(isinstance(root,Product) or isinstance(root, Sum)):
        for child in root.children:
            print_spn(child)

def print_spn_based_on_scope(root):
    if(isinstance(root, Leaf)):
        print("LEAF: ", root,"with scope:",root.scope, " with parameters:",  [root.parameters[x] for x in root.scope])
    elif(isinstance(root, Sum)):
        print("SUM: ", root, "with weights:", root.weights)
    elif(isinstance(root, Product)):
        print("PRODUCT: ", root, " scope: ", root.scope)

    if(isinstance(root,Product) or isinstance(root, Sum)):
        for child in root.children:
            print_spn_based_on_scope(child)

def save_spn_scopes_to_file(root, file):
    if (isinstance(root, Leaf)):
        try:
            file.write(str(root) + ' scope: ' + str(root.scope) + '\n')
        except:
            print("failed to write to file")

    if (isinstance(root,Product) or isinstance(root, Sum)):
        for child in root.children:
            save_spn_scopes_to_file(child,file)


def print_spn_scopes(root):
    
    if(isinstance(root, Leaf)):
        print("LEAF: ", root, "with scope: ", root.new_scope)
    elif(isinstance(root, Sum)):
        print("SUM: ", root, "with scope: ", root.new_scope)
    elif(isinstance(root, Product)):
        print("PRODUCT: ", root, "with scope: ", root.new_scope)

    if(isinstance(root,Product) or isinstance(root, Sum)):
        for child in root.children:
            print_spn_scopes(child)

def save_sum_weights(spn):
    sums = []
    get_all_sums(spn, sums)
    number_of_bins = 20
    label = "edit2"

    for s in sums:
        weights = s.markov_chain
        """
        plt.figure(np.random.randint(0, 1000000000000))
        first = [i[0] for i in weights]
        second = [i[1] for i in weights]
        plt.hist(first, bins=number_of_bins)
        plt.savefig(fname="Sum_weights_first_{}".format(s.name))

        plt.figure(np.random.randint(0, 1000000000000))
        indexes = list(range(0, len(first)))
        plt.plot(indexes, first)
        plt.savefig(fname="Sum_weights_plot_first_{}".format(s.name))


        plt.figure(np.random.randint(0, 1000000000000))

        plt.hist(second, bins=number_of_bins)
        plt.savefig(fname="Sum_weights_second_{}".format(s.name))
        """
        file_name = "samples_" + s.name + "_weights_" + label + ".txt"

        f = open(file_name, "w+")
        for i in weights:
            first = i[0]
            second = i[1]
            try:
                f.write(str(first) + ' ' + str(second) + '\n')
            except:
                print("failed to write to file")

        f.close() 


def save_leaf_scopes(spn):
    leaves = []
    get_all_leaves(spn, leaves)
    label = "edit2"

    for leaf in leaves:
        file_name = "scopes_" + leaf.name + label + ".txt"
        scopes = leaf.scope_store

        f = open(file_name, "w+")
        for scope in scopes:
            try:
                f.write(str(scope) + '\n')
            except:
                print("failed to write to file")

        f.close() 


def save_likelihood_weights(spn):

    leaves = []
    get_all_leaves(spn, leaves)
    label = "edit2"
    for leaf in leaves:
        file_name = "likelihoodweights_" + leaf.name + label + ".txt"
        weights = leaf.all_likelihood_weights

        f = open(file_name, "w+")
        
        try:
            f.write(str(weights) + '\n')
        except:
            print("failed to write to file")

        f.close() 

def set_leaf_data_size(spn, dims):
    leaves = []
    get_all_leaves(spn, leaves)

    for leaf in leaves:
        for dim in range(dims):
            for likelihood in leaf.likelihoods:
                if(likelihood == "Gaussian"):
                    gaussian_data = leaf.data[dim]['Gaussian']

                    leaf.data_size[dim]['Gaussian'].append(len(gaussian_data))
                elif( likelihood == "Poisson"):
                    poisson_data = leaf.data[dim]['Poisson']

                    leaf.data_size[dim]['Poisson'].append(len(poisson_data))
                elif(likelihood == "Gamma"):
                    gamma_data = leaf.data[dim]['Gamma']

                    leaf.data_size[dim]['Gamma'].append(len(gamma_data))

                elif likelihood == "Bernoulli":
                    bernoulli_data = leaf.data[dim]['Bernoulli']

                    leaf.data_size[dim]['Bernoulli'].append(len(bernoulli_data))

                elif likelihood == "Geometric":
                    geometric_data = leaf.data[dim]['Geometric']

                    leaf.data_size[dim]['Geometric'].append(len(geometric_data))
                
                elif likelihood == "Exponential":
                    exponential_data = leaf.data[dim]['Exponential']

                    leaf.data_size[dim]['Exponential'].append(len(exponential_data))


def save_leaf_parameters(spn, dims):
    leaves = []
    get_all_leaves(spn, leaves)
    number_of_bins = 20
    burn_in = 0
    label = "edit2"

    for leaf in leaves:

        for dim in range(dims):
            all_parameters = leaf.markov_chain[dim]

            for likelihood in leaf.likelihoods:
                if(likelihood == "Gaussian"):
                    gaussian_parameters = all_parameters['Gaussian'][burn_in:]
                    gaussian_data = leaf.data[dim]['Gaussian']
                    gaussian_data_size = leaf.data_size[dim]['Gaussian']
                    """
                    means_gaussian = [d['mean'] for d in gaussian_parameters]
                    stdevs_gaussian = [d['stdev'] for d in gaussian_parameters]

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(means_gaussian, bins=number_of_bins)
                    plt.title("dim: {}, Gaussian means for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_gaussian_means_leaf_{}".format(dim, leaf.name))

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(stdevs_gaussian, bins=number_of_bins)
                    plt.title("dim: {}, Gaussian stdevs for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_gaussian_stdevs_leaf_{}".format(dim, leaf.name))

                    """

                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_gaussian_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in gaussian_parameters:
                        mean = i["mean"]
                        stdev = i['stdev']
                        try:
                            f.write(str(mean) + ' ' + str(stdev) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_gaussian_" + label + ".txt"

                    f = open(file_name, "w+")
                    for size in gaussian_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """

                elif(likelihood == "Poisson"):
                    poisson_parameters = all_parameters['Poisson'][burn_in:]
                    means_poisson = [d['mean'] for d in poisson_parameters]
                    poisson_data = leaf.data[dim]['Poisson']
                    poisson_data_size = leaf.data_size[dim]['Poisson']
                    """
                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(means_poisson, bins=number_of_bins)
                    plt.title("dim: {}, Poisson means for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_poisson_means_leaf_{}".format(dim, leaf.name))
                    """

                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_poisson_without_index_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in poisson_parameters:
                        mean = i["mean"]
                        try:
                            f.write(str(mean) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_poisson_without_index_" + label + ".txt"

                    f = open(file_name, "w+")
                    for size in poisson_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """

                elif(likelihood == "Gamma"):
                    gamma_parameters = all_parameters['Gamma'][burn_in:]
                    gamma_data = leaf.data[dim]['Gamma']
                    gamma_data_size = leaf.data_size[dim]['Gamma']

                    """a_gamma = [d['a'] for d in gamma_parameters]
                    b_gamma = [d['b'] for d in gamma_parameters]

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(a_gamma, bins=number_of_bins)
                    plt.title("dim: {}, Gamma alpha for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_gamma_alphas_leaf_{}".format(dim, leaf.name))

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(b_gamma, bins=number_of_bins)
                    plt.title("dim: {}, Gamma b for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_gamma_betas_leaf_{}".format(dim, leaf.name))
                    """

                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_gamma_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in gamma_parameters:
                        a = i["a"]
                        b = i["b"]
                        try:
                            f.write(str(a) + ' ' + str(b) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 

                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_gamma_" + label + ".txt"
                    f = open(file_name, "w+")
                    for size in gamma_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """
                elif(likelihood == "Bernoulli"):
                    bernoulli_parameters = all_parameters['Bernoulli'][burn_in:]
                    bernoulli_data = leaf.data[dim]['Bernoulli']
                    bernoulli_data_size = leaf.data_size[dim]['Bernoulli']

                    """p_bernoulli = [d['p'] for d in bernoulli_parameters]

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(p_bernoulli, bins=number_of_bins)
                    plt.title("dim: {}, Bernoulli p for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_bernoulli_p_leaf_{}".format(dim, leaf.name))
                    """
                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_bernoulli_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in bernoulli_parameters:
                        p = i["p"]
                        try:
                            f.write(str(p) + '\n')
                        except:
                            print("failed to write to file")

                    f.close() 
                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_bernoulli_" + label + ".txt"
                    f = open(file_name, "w+")
                    for size in bernoulli_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close()
                    """
                elif(likelihood == "Geometric"):
                    geometric_parameters = all_parameters['Geometric'][burn_in:]
                    geometric_data = leaf.data[dim]['Geometric']
                    geometric_data_size = leaf.data_size[dim]['Geometric']

                    """p_geometric = [d['p'] for d in geometric_parameters]

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(p_geometric, bins=number_of_bins)
                    plt.title("dim: {}, Geometric p for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_geometric_p_leaf_{}".format(dim, leaf.name))
                    """
                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_geometric_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in geometric_parameters:
                        p = i["p"]
                        try:
                            f.write(str(p) + '\n')
                        except:
                            print("failed to write to file")

                    f.close()
                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_geometric_" + label + ".txt"
                    f = open(file_name, "w+")
                    for size in geometric_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close()
                    """
                elif(likelihood == "Exponential"):
                    exponential_parameters = all_parameters['Exponential'][burn_in:]
                    exponential_data = leaf.data[dim]['Exponential']
                    exponential_data_size = leaf.data_size[dim]['Exponential']

                    """l_exponential = [d['l'] for d in exponential_parameters]

                    plt.figure(np.random.randint(0, 1000000000000))
                    plt.hist(l_exponential, bins=number_of_bins)
                    plt.title("dim: {}, Exponential l for leaf: {}".format(dim, leaf.name))
                    plt.savefig(fname="Dim{}_exponential_l_leaf_{}".format(dim, leaf.name))
                    """
                    file_name = "samples_" + leaf.name + "_dim" + str(dim) + "_exponential_" + label + ".txt"

                    f = open(file_name, "w+")
                    for i in exponential_parameters:
                        l = i["l"]
                        try:
                            f.write(str(l) + '\n')
                        except:
                            print("failed to write to file")

                    f.close()

                    """
                    file_name = "datasize_" + leaf.name + "_dim" + str(dim) + "_exponential_" + label + ".txt"
                    f = open(file_name, "w+")
                    for size in exponential_data_size:
                        try:
                            f.write(str(size) + '\n')
                        except:
                            print("failed to write to file")

                    f.close()
                    """

def save_sum_mh_prob(spn):
    sums = []
    get_all_sums(spn, sums)
    label = "edit2"
    

    for s in sums:
        mh_prob = s.mh_prob_w

        file_name = "mh_probs_w_" + s.name + "_"+ label + ".txt"

        f = open(file_name, "w+")
        for i in mh_prob:

            try:
                f.write(str(i) + '\n')
            except:
                print("failed to write to file")

        f.close() 

        mh_prob_all = s.mh_prob_w_all

        file_name = "mh_probs_w_all_" + s.name + "_"+ label + ".txt"

        f = open(file_name, "w+")
        for i in mh_prob_all:

            try:
                f.write(str(i) + '\n')
            except:
                print("failed to write to file")

        f.close() 


        mh_prob_z = s.mh_prob_z
        file_name = "mh_probs_acceptance_" + s.name + "_"+ label + ".txt"

        f = open(file_name, "w+")
        for i in mh_prob_z:

            try:
                f.write(str(i) + '\n')
            except:
                print("failed to write to file")

        f.close() 


