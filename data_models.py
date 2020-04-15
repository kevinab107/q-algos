import numpy as np
from networkx import nx


def create_tsp_data_model(num_nodes=5):

    np.random.seed(0)
    xy = np.random.randint(100, 1000, (2, num_nodes))

    # compute distance matrix
    dx = np.subtract.outer(xy[0], xy[0])
    dy = np.subtract.outer(xy[1], xy[1])
    dist = np.sqrt(dx ** 2 + dy ** 2)
    data = {}
    data["distance_matrix"] = dist.tolist()
    data["distance_matrix_numpy"] = dist
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data

def create_tsp_edge_data_model(num_nodes=5):


    # compute distance mat
    data["distance_matrix"] = dist.tolist()
    data["distance_matrix_numpy"] = dist
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def get_maxcut_data_model():
    """ graph to deal with ibmq hardwares"""
    n = 5
    V = np.arange(0, n, 1)
    E = [(0, 1, 3.0), (1, 2, 2.0), (2, 3, 2.0), (3, 4, 3.0), (4, 0, 1.0), (0, 3, 3.0)]

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G


def get_max_cut_star_5():
    n = 5
    V = np.arange(0, n, 1)
    E = [
        (0, 1, 3.0),
        (1, 2, 2.0),
        (2, 3, 2.0),
        (3, 4, 3.0),
        (4, 0, 1.0),
        (0, 3, 3.0),
        (1, 0, 3.0),
        (2, 1, 2.0),
        (3, 2, 2.0),
        (4, 3, 3.0),
        (0, 4, 1.0),
        (3, 0, 3.0),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G
