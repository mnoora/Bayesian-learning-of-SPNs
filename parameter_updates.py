from scipy.stats import gamma, norm, invgamma, beta
import numpy as np

def update_hyperparameters(leaf, dim):
    for l in leaf.likelihoods:
        update_hyperparameters_directions(leaf, leaf.data[dim][l], l, dim)
       

def update_hyperparameters_directions(leaf, data, likelihood, dim):

    if(likelihood == "Gaussian"):
        update_hyperparameters_for_gaussian_leaf(leaf, data, dim)
    elif(likelihood == "Exponential"):
        update_hyperparameters_for_exponential_leaf(leaf, data, dim)
    elif(likelihood == "Poisson"):
        update_hyperparameters_for_poisson_leaf(leaf, data, dim)
    elif(likelihood == "Gamma"):
        update_hyperparameters_for_gamma_leaf(leaf, data, dim)
    elif(likelihood == "Bernoulli"):
        update_hyperparameters_for_bernoulli_leaf(leaf, data, dim)
    elif(likelihood == "Geometric"):
        update_hyperparameters_for_geometric_leaf(leaf, data, dim)


def update_hyperparameters_for_gaussian_leaf(leaf, data, dim):
    # Prior is Normal-Inverse-Gamma
    X = np.array(data)
    
    N = len(X)
    if(N==0):
        return

    x_hat = np.mean(X)
    x_sum = np.sum(X)
    x_sdiff = np.sum((X - x_hat)**2)

    prior = leaf.prior[dim]["Normal-Inverse-Gamma"]
    inv_v = 1.0 / prior["v"]
    v_n = 1 / (inv_v + N)

    mu = (prior["mu"]*prior["v"] + x_sum) *v_n
    v = prior["v"] + N
    a = prior["a"] + N / 2
    b = prior["b"] + x_sdiff / 2 + ((N * prior["v"]) / (prior["v"] + N)) * (((x_hat - prior["mu"])**2 / 2))

    leaf.hyperparameters[dim]["Gaussian"] = {"mu": mu,
                                             "v": v,
                                             "a": a,
                                             "b": b}
    
    sigma2_sample = invgamma.rvs(a=a, size=1) * b
    std = np.sqrt(sigma2_sample *v_n )
    mu_sample = norm.rvs(size =1, loc=mu, scale=std)

    leaf.parameters[dim]["Gaussian"] = {"mean": mu_sample[0],
                                        "stdev": np.sqrt(sigma2_sample)[0]}

    leaf.markov_chain[dim]["Gaussian"].append({"mean": mu_sample[0],
                                        "stdev": np.sqrt(sigma2_sample)[0]})


def update_hyperparameters_for_poisson_leaf(leaf, data, dim):
    # Prior is gamma
    X = np.array(data)

    N = len(X)

    x_sum = np.sum(X)

    prior = leaf.prior[dim]["Gamma"]

    a = prior["a"] + x_sum
    b = prior["b"] + N
    if(a<0):
        return 

    leaf.hyperparameters[dim]["Poisson"] = {"a": a,
                                            "b": b}
    
    lambda_sample = gamma.rvs(size=1, a=a, scale=1.0 / b)

    leaf.parameters[dim]["Poisson"] = {"mean": lambda_sample[0]}

    leaf.markov_chain[dim]["Poisson"].append({"mean": lambda_sample[0]})

   


#with fixed alpha
def update_hyperparameters_for_gamma_leaf(leaf, data, dim):
    # Prior is gamma
    X = np.array(data)
    N = len(X)

    x_sum = np.sum(X)

    prior = leaf.prior[dim]["Gamma"]

    a = prior["a"] + N*1
    b = prior["b"] + x_sum

    if(a<=0 or b <=0):
        return

    leaf.hyperparameters[dim]["Gamma"] = {"a": a,
                                          "b": b}

    rate_sample = gamma.rvs(size=1, a=a, scale=1.0/b)

    leaf.parameters[dim]["Gamma"] = {"a": 1,
                                     "b": rate_sample[0]}

    leaf.markov_chain[dim]["Gamma"].append({"a": 1,
                                     "b": rate_sample[0]})



def update_hyperparameters_for_bernoulli_leaf(leaf, data, dim):
    # Prior is beta
    X = np.array(data)
    N = len(X)

    x_sum = np.sum(X)

    prior = leaf.prior[dim]["Beta"]

    a = x_sum + prior["a"]
    b = (N - x_sum) + prior["b"]
    if (a <= 0 or b <= 0):
        return

    leaf.hyperparameters[dim]["Bernoulli"] = {"a": a,
                                              "b": b}

    p_sample = beta.rvs(a, b)

    leaf.parameters[dim]["Bernoulli"] = {"p": p_sample}

    leaf.markov_chain[dim]["Bernoulli"].append({"p": p_sample})


def update_hyperparameters_for_geometric_leaf(leaf, data, dim):
    # Prior is beta
    X = np.array(data)
    N = len(X)

    x_sum = np.sum(X)

    prior = leaf.prior[dim]["Beta"]

    a = N + prior["a"]
    b = prior["b"] + x_sum -N

    if (a <= 0 or b <= 0):
        return

    leaf.hyperparameters[dim]["Geometric"] = {"a": a,
                                              "b": b}

    p_sample = beta.rvs(a,b, size=1)

    leaf.parameters[dim]["Geometric"] = {"p": p_sample[0]}

    leaf.markov_chain[dim]["Geometric"].append({"p": p_sample[0]})

def update_hyperparameters_for_exponential_leaf(leaf, data, dim):
    # Prior is gamma
    X = np.array(data)
    N = len(X)

    x_sum = np.sum(X)

    prior = leaf.prior[dim]["Gamma"]

    a = N + prior["a"]
    b = prior["b"] + x_sum

    if (a <= 0 or b <= 0):
        return


    leaf.hyperparameters[dim]["Exponential"] = {"a": a,
                                                "b": b}

    lambda_sample = gamma.rvs(size=1, a=a, scale=1.0/b)

    leaf.parameters[dim]["Exponential"] = {"l": lambda_sample[0]}

    leaf.markov_chain[dim]["Exponential"].append({"l": lambda_sample[0]})