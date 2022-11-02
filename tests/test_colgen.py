import cvrplib

from scip_routing.solver import VRPTWSolver
from scip_routing.utils import minify_instance, instance_graph


def test_finds_optimal():
    instance, sol = cvrplib.download('R101', solution=True)

    instance = minify_instance(instance, 20)

    graph = instance_graph(instance)

    solver = VRPTWSolver(graph=graph,
                         instance=instance,
                         verbose=True)

    solver.solve()

    assert solver.rmp.getObjVal() == 742
