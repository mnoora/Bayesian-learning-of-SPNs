from generate_data import create_generating_spn, generate_data_from_example_SPN
from node import Product, Sum, Leaf
from spn_leaves import get_all_leaves, get_all_nodes
from structure_learning import PDF_for_likelihood_model
import numpy as np
from spn.algorithms.Inference import log_likelihood


def calculate_probability(spn, data_point):
    root_node = spn
    leaves = []
    get_all_leaves(spn, leaves)
    node_probs = {}

    for leaf in leaves:
        prob = 1
        for dim in leaf.scope:
            
            params = leaf.parameters[dim]
            likelihood_weights = leaf.likelihood_weights[dim]
            i = 0
            for key, values in params.items():               
                weights = likelihood_weights
                prob_density = weights[i] * PDF_for_likelihood_model(key, data_point[dim], params)
                if (prob_density != 0):
                    prob *= weights[i] * PDF_for_likelihood_model(key, data_point[dim], params)
                i+=1

        node_probs[leaf] = prob

    parents = []
    for leaf in leaves:
        parent = leaf.parent
        parents.append(parent.parent)
        node_probs = calculate_prob(parent, node_probs)

    parents_new = []
    for node in parents:
        node_probs = calculate_prob(node, node_probs)
        parents_new.append(node.parent)
    
    root = 1
    for node in parents_new:
        node_probs = calculate_prob(node, node_probs)
        root = node.parent

    node_probs = calculate_prob(root, node_probs)
    return node_probs[root_node]



def calculate_prob(node, node_probs):
    children = node.children

    if isinstance(node, Product):
        prob = 1
        for child in children:
            prob *= node_probs[child]
    elif isinstance(node, Sum):
        i = 0
        prob = 0
        for child in children:
            prob += node.weights[i]*node_probs[child]
            i += 1
    node_probs[node] = prob

    return node_probs



def create_SPN():
    leaf_1 = Leaf(dims=4,scope=[1,3], likelihood_weights= {0: [0.04471504, 0.15094893, 0.06411904, 0.5731136 , 0.11966964,
       0.04743375], 1: [0.39115097, 0.04453082, 0.0346668 , 0.02655084, 0.08932513,
       0.41377545], 2: [0.16164837, 0.16632662, 0.06222008, 0.16238497, 0.03754004,
       0.40987993], 3: [0.16766329, 0.33531921, 0.14068287, 0.15028568, 0.19370654,
       0.0123424 ]})
    leaf_2 = Leaf(dims=4,scope=[0], likelihood_weights={0: [0.18042725, 0.2793627 , 0.23288394, 0.2525804 , 0.02019807,
       0.03454763], 1: [0.06915918, 0.24256826, 0.013003  , 0.35826004, 0.06629767,
       0.25071185], 2: [0.01734041, 0.02167949, 0.27841023, 0.22111796, 0.35967022,
       0.10178169], 3: [0.11699247, 0.22900064, 0.08452643, 0.30799278, 0.22859384,
       0.03289384]})
    leaf_3 = Leaf(dims=4,scope=[0], likelihood_weights={0: [0.09123426, 0.39261332, 0.15499768, 0.09602062, 0.08474741,
       0.18038671], 1: [0.27676415, 0.30896814, 0.06572794, 0.16338798, 0.04541661,
       0.13973519], 2: [0.41777472, 0.05018235, 0.19384989, 0.15246197, 0.03275082,
       0.15298024], 3: [0.03970764, 0.29875201, 0.22711952, 0.08111804, 0.27510165,
       0.07820114]})
    leaf_4 = Leaf(dims=4,scope=[1,3], likelihood_weights= {0: [0.00064588, 0.19328318, 0.06848956, 0.10933702, 0.17341915,
       0.4548252 ], 1: [0.18567947, 0.00225131, 0.15940509, 0.37060914, 0.20804113,
       0.07401386], 2: [0.29692921, 0.14758366, 0.26702133, 0.04682024, 0.20600769,
       0.03563786], 3: [0.04907342, 0.01802554, 0.02271254, 0.86019683, 0.00494076,
       0.0450509 ]})

    leaf_5 = Leaf(dims=4,scope=[], likelihood_weights= {0: [0.12081697, 0.27805842, 0.05566536, 0.43220544, 0.01980926,
       0.09344455], 1: [0.4391102 , 0.06856449, 0.10040055, 0.03373058, 0.04268266,
       0.31551152], 2: [0.10200533, 0.11228054, 0.17607304, 0.28449746, 0.03523257,
       0.28991106], 3: [0.49434881, 0.04861398, 0.0420915 , 0.02366059, 0.17087954,
       0.22040559]})
    leaf_6 = Leaf(dims=4,scope=[2], likelihood_weights={0: [0.12317119, 0.1498635 , 0.27219366, 0.21756819, 0.00709099,
       0.23011247], 1: [0.09827959, 0.15008582, 0.28422641, 0.18399448, 0.0414932 ,
       0.24192051], 2: [0.08186627, 0.19758733, 0.13031045, 0.41676931, 0.05775452,
       0.11571213], 3: [0.22177424, 0.10115934, 0.02456517, 0.05723339, 0.41862558,
       0.17664228]})
    leaf_7 = Leaf(dims=4,scope=[], likelihood_weights={0: [0.54626916, 0.04504935, 0.02611212, 0.06313812, 0.06675865,
       0.25267261], 1: [0.00845917, 0.08500397, 0.06514102, 0.43211837, 0.14721968,
       0.2620578 ], 2: [0.48046616, 0.06326602, 0.0638412 , 0.04226897, 0.25452758,
       0.09563008], 3: [0.32645944, 0.27914413, 0.09452016, 0.19258036, 0.09682918,
       0.01046673]})
    leaf_8 = Leaf(dims=4,scope=[2], likelihood_weights={0:[0.14217429, 0.39266936, 0.02250383, 0.00926297, 0.04120518,
       0.39218438], 1: [0.1883948 , 0.35449012, 0.24758927, 0.02813966, 0.05969489,
       0.12169128], 2: [0.02340634, 0.0140431 , 0.51507126, 0.00840383, 0.30771994,
       0.13135552], 3: [0.27114143, 0.0140775 , 0.0350278 , 0.03298477, 0.41590585,
       0.23086265]})

    leaf_9 = Leaf(dims=4,scope=[], likelihood_weights={0: [0.12894392, 0.13792806, 0.46046104, 0.08352563, 0.12405014,
       0.0650912 ], 1: [0.37454475, 0.02453764, 0.01316413, 0.21487963, 0.32033675,
       0.05253711], 2: [0.11054623, 0.13078287, 0.1119538 , 0.20237859, 0.08728392,
       0.3570546 ], 3: [0.16189578, 0.01543066, 0.11513944, 0.40874786, 0.2773215 ,
       0.02146476]})
    leaf_10 = Leaf(dims=4,scope=[0,1,2], likelihood_weights={0: [0.03617924, 0.07815588, 0.52509155, 0.14124522, 0.01706963,
       0.20225848], 1: [0.59110308, 0.07906662, 0.19087516, 0.02461863, 0.09139953,
       0.02293699], 2: [0.12062799, 0.25709988, 0.32015628, 0.11449979, 0.10259345,
       0.0850226 ], 3: [0.55198778, 0.03756076, 0.27816507, 0.0162277 , 0.09608958,
       0.0199691 ]})
    leaf_11 = Leaf(dims=4,scope=[0,1], likelihood_weights={0: [0.26087489, 0.26977722, 0.45454239, 0.00334351, 0.00319302,
       0.00826896], 1: [0.01964854, 0.1783074 , 0.24497904, 0.32065493, 0.21229377,
       0.02411633], 2: [0.02753987, 0.04103113, 0.21076823, 0.08715794, 0.19086007,
       0.44264276], 3: [0.1440971 , 0.02026774, 0.09315793, 0.02276278, 0.42969518,
       0.29001927]})
    leaf_12 = Leaf(dims=4,scope=[2], likelihood_weights={0: [0.41171737, 0.32310931, 0.09406527, 0.02320861, 0.1001892 ,
       0.04771026], 1: [0.25103773, 0.1370167 , 0.14345758, 0.18989918, 0.15630914,
       0.12227968], 2: [0.07218312, 0.35083065, 0.02273985, 0.15210574, 0.13075966,
       0.27138099], 3: [0.04165826, 0.13747729, 0.0289803 , 0.38572729, 0.27081383,
       0.13534303]})

    leaf_13 = Leaf(dims=4,scope=[], likelihood_weights={0: [0.04916988, 0.25146409, 0.20674723, 0.27694806, 0.14614378,
       0.06952696], 1: [0.06602861, 0.03608312, 0.00431523, 0.41658917, 0.29121194,
       0.18577194], 2: [1.69523236e-01, 2.18502341e-02, 1.65445480e-01, 2.54188699e-01,
       3.88695072e-01, 2.97278949e-04], 3: [0.1130676 , 0.0126015 , 0.18522735, 0.00786204, 0.39732358,
       0.28391794]})
    leaf_14 = Leaf(dims=4,scope=[3], likelihood_weights={0: [0.29746053, 0.12955128, 0.12235246, 0.02375948, 0.41555249,
       0.01132375], 1: [0.04565745, 0.2200305 , 0.04877825, 0.19488815, 0.28822793,
       0.20241772], 2: [0.26289965, 0.09890529, 0.25488323, 0.19685282, 0.04405738,
       0.14240162], 3: [0.06351042, 0.2203901 , 0.00265932, 0.28044954, 0.19594142,
       0.23704921]})
    leaf_15 = Leaf(dims=4,scope=[], likelihood_weights={0: [0.06796923, 0.32753688, 0.07447423, 0.06128741, 0.05749793,
       0.41123433], 1: [0.21342405, 0.15687685, 0.10404235, 0.08011675, 0.23374894,
       0.21179107], 2: [0.00069121, 0.31078262, 0.21409739, 0.07789462, 0.23016647,
       0.16636769], 3: [0.29248742, 0.09246912, 0.05089566, 0.01618301, 0.25614151,
       0.29182327]})
    leaf_16 = Leaf(dims=4,scope=[3], likelihood_weights={0: [0.33752389, 0.00171395, 0.14207931, 0.10693566, 0.3593271 ,
       0.05242009], 1: [0.27801958, 0.34979624, 0.10824197, 0.13857278, 0.10153189,
       0.02383753], 2: [0.13195537, 0.39266753, 0.08001724, 0.1581078 , 0.11046047,
       0.12679159], 3: [0.1379414 , 0.01451487, 0.0608268 , 0.49858075, 0.23381169,
       0.05432449]})

    product_1 = Product(children=[leaf_1, leaf_2],scope=[0,1,3])
    product_2 = Product(children=[leaf_3, leaf_4],scope=[0,1,3])
    product_3 = Product(children=[leaf_5, leaf_6],scope=[2])
    product_4 = Product(children=[leaf_7, leaf_8],scope=[2])

    product_5 = Product(children=[leaf_9, leaf_10],scope=[0,1,2])
    product_6 = Product(children=[leaf_11, leaf_12],scope=[0,1,2])
    product_7 = Product(children=[leaf_13, leaf_14],scope=[3])
    product_8 = Product(children=[leaf_15, leaf_16],scope=[3])

    sum_1 = Sum(children=[product_1,product_2],scope=[0,1,3], weights=[0.35, 0.65])
    sum_2 = Sum(children=[product_3,product_4],scope=[2], weights=[0.84, 0.16])

    sum_3 = Sum(children=[product_5,product_6],scope=[0,1,2], weights=[0.09, 0.91])
    sum_4 = Sum(children=[product_7,product_8],scope=[3], weights=[0.51, 0.49])

    product_1_first_layer = Product(children=[sum_1, sum_2],scope=[0,1,2,3])
    product_2_first_layer = Product(children=[sum_3, sum_4],scope=[0,1,2,3])

    root = Sum(children=[product_1_first_layer,product_2_first_layer],scope=[0,1,2,3], weights=[0.58, 0.42])

    return root

def assign_parameters(root):
    leaves = []
    get_all_leaves(root, leaves)
    dims = 4

    for leaf in leaves:
        for dim in range(dims):
            params = leaf.parameters[dim]

            for key, _ in params.items():
                model = key
                node = leaf

                if model == "Poisson":
                    model = "_poisson_without_index"
                elif model == "Gamma":
                    model = "_gamma"
                elif model == "Gaussian":
                    model = "_gaussian"
                elif model == "Exponential":
                    model = "_exponential"
                elif model == "Geometric":
                    model = "_geometric"
                elif model == "Bernoulli":
                    model = "_bernoulli"

                file_name = "gibbs1/samples_" + str(node) + "_dim" + str(dim) + model + "_gibbs_edit1.txt"
                with open(file_name, 'r') as f:
                    lines = f.readlines()
                    
                    if lines:
                        last_line = lines[-1]
                       
                        if key == "Poisson":
                            leaf.parameters[dim][key] =  {"mean": float(last_line)}
                        elif key == "Gamma":
                            p = last_line.split()
                            leaf.parameters[dim][key] =  {"a": float(p[0]), "b": float(p[1])}
                        elif key == "Gaussian":
                            p = last_line.split()
                            leaf.parameters[dim][key] = {"mean": float(p[0]), "stdev": float(p[1])}
                        elif key == "Exponential":
                            leaf.parameters[dim][key] = {"l": float(last_line)}
                        elif key == "Geometric":
                            leaf.parameters[dim][key] = {"p": float(last_line)}
                        elif key == "Bernoulli":
                            leaf.parameters[dim][key] = {"p": float(last_line)}
    return root


def assign_parents(spn):
    nodes = []
    get_all_nodes(spn, nodes)

    for node in nodes:
        if isinstance(node, Sum) or isinstance(node, Product):
            for child in node.children:
                child.parent = node

    return spn

def average(lst):
    return sum(lst) / len(lst)

def calculate_all_prob():
    data = generate_data_from_example_SPN(9000)

    test_dataset_size = 2000

    test_data = data[-test_dataset_size:]

    spn = create_SPN()
    spn = assign_parameters(spn)
    spn = assign_parents(spn)

    likelihoods = []

    for point in test_data:
        likelihoods.append(np.log(calculate_probability(spn,point)))

    print(average(likelihoods))


    generating_spn = create_generating_spn()

    ll = log_likelihood(generating_spn, test_data)
    print(average(ll))
calculate_all_prob()