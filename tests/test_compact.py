import cvrplib

from scip_routing.utils import minify_instance, instance_graph
from scip_routing.compact import solve

def test_finds_optimal():
    instance, sol = cvrplib.download('R101', solution=True)

    instance = minify_instance(instance, 10)

    graph = instance_graph(instance)

    solve(graph, instance, 5)    