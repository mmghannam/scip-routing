from ast import literal_eval as make_tuple
from functools import lru_cache

import networkx as nx
# from vrplib.Instance import VRPTW

class Instance:
    pass

def instance_graph(given_instance):
    graph = nx.DiGraph()
    instance = Instance()
    instance.n_customers = len(given_instance['demand']) - 1
    instance.customers = list(range(1, instance.n_customers + 1))
    instance.distances = given_instance['edge_weight'][0]
    instance.earliest = [x[0] for x in given_instance['time_window']]
    instance.latest = [x[1] for x in given_instance['time_window']]
    instance.depot = 0
    instance.demands = list(given_instance['demand'])
    instance.service_times = list(given_instance['service_time'])
    instance.capacity = given_instance['capacity']
    instance.n_vehicles = given_instance['vehicles']

    # edges between customers
    for ci in instance.customers:
        for cj in instance.customers:
            if ci != cj:
                graph.add_edge(ci, cj, distance=instance.distances[ci][cj])

    # edges from start depot
    for c in instance.customers:
        graph.add_edge(instance.depot, c, distance=instance.distances[instance.depot][c])

    # edges to end depot
    end_depot = instance.n_customers + 1
    for c in instance.customers:
        graph.add_edge(c, end_depot, distance=instance.distances[c][instance.depot])
    graph.add_edge(instance.depot, end_depot, distance=0)

    return instance, graph


def minify_instance(instance, only_first):
    distances = [[0] * (only_first + 1) for _ in range(only_first + 1)]
    for i in range(only_first + 1):
        for j in range(only_first + 1):
            distances[i][j] = int(instance["edge_weight"][i][j])
    
    instance["edge_weight"]=distances,
    instance["demand"]=instance["demand"][:only_first+1]
    instance["service_time"]=instance["service_time"][:only_first+1]


def var_to_edges(var):
    return var_name_to_edges(str(var))


@lru_cache(maxsize=None)
def var_name_to_edges(var_name):
    var_name = var_name if var_name[0] != "t" else var_name[2:]
    nodes = list(make_tuple(var_name))
    nodes[-1] = nodes[0]
    return set(zip(nodes[:-1], nodes[1:]))
