class NormalInverseGamma:
    def __init__(self, mu, v, a, b):
        self.mu = mu
        self.v = v
        self.a = a
        self.b = b

    def __repr__(self):
        return "Parameters mu={} v={} a={} b={}".format(self.mu,self.v,self.a,self.b)

class Gamma:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Normal:
    def __init__(self, sd, var):
        self.sd = sd
        self.var = var

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Dirichlet:
    def __init__(self, alphas):
        self.alphas = alphas