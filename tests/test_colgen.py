import cvrplib

from scip_routing.solver import VRPTWSolver
from scip_routing.utils import minify_instance, instance_graph


def test_finds_optimal():
    instance, sol = cvrplib.download('R101', solution=True)

    instance = minify_instance(instance, 50)

    graph = instance_graph(instance)

    py_solver = VRPTWSolver(graph=graph,
                         instance=instance,
                         verbosity=1,
                         pricing_strategy="py")
    py_solver.solve()
    rust_solver = VRPTWSolver(graph=graph,
                         instance=instance,
                         verbosity=1,
                         pricing_strategy="rust")
    rust_solver.solve()

    assert py_solver.rmp.getObjVal() == rust_solver.rmp.getObjVal() == 1031
