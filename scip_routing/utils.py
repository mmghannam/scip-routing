from cvrplib.Instance import VRPTW
import networkx as nx


def instance_graph(instance: VRPTW):
    graph = nx.DiGraph()

    # edges between customers
    for ci in instance.customers:
        for cj in instance.customers:
            graph.add_edge(ci, cj, distance=instance.distances[ci][cj])

    # edges from start depot
    for c in instance.customers:
        graph.add_edge(instance.depot, c, distance=instance.distances[instance.depot][c])

    # edges to end depot
    end_depot = instance.n_customers + 1
    for c in instance.customers:
        graph.add_edge(c, end_depot, distance=instance.distances[c][instance.depot])

    return graph


def minify_instance(instance, only_first):
    instance.customers = instance.customers[:only_first]
    instance.earliest = instance.earliest[:only_first + 1]
    instance.latest = instance.latest[:only_first + 1]
    instance.demands = instance.demands[:only_first + 1]
    instance.n_customers = only_first
    return instance
