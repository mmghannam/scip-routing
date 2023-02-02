from ast import literal_eval as make_tuple
from functools import lru_cache

import networkx as nx
from cvrplib.Instance import VRPTW


def instance_graph(instance: VRPTW):
    graph = nx.DiGraph()

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

    return graph


def minify_instance(instance, only_first):
    distances = [[0] * (only_first + 1) for _ in range(only_first + 1)]
    for i in range(only_first + 1):
        for j in range(only_first + 1):
            distances[i][j] = instance.distances[i][j]
    return VRPTW(
        n_vehicles=instance.n_vehicles,
        earliest=instance.earliest[:only_first + 1],
        latest=instance.latest[:only_first + 1],
        name=instance.name,
        dimension=instance.dimension,
        n_customers=only_first,
        depot= instance.depot,
        customers=instance.customers[:only_first],
        capacity=instance.capacity,
        distances=distances,
        demands=instance.demands[:only_first+1],
        service_times=instance.service_times[:only_first+1],
        coordinates=instance.coordinates[:only_first+1]
    )


def var_to_edges(var):
    return var_name_to_edges(str(var))


@lru_cache(maxsize=None)
def var_name_to_edges(var_name):
    var_name = var_name if var_name[0] != "t" else var_name[2:]
    nodes = list(make_tuple(var_name))
    nodes[-1] = nodes[0]
    return set(zip(nodes[:-1], nodes[1:]))
