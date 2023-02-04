import cvrplib

from scip_routing.compact import solve_compact
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


def test_same_answer_as_compact():
    instance, sol = cvrplib.download('R101', solution=True)
    instance = minify_instance(instance, 10)
    graph = instance_graph(instance)
    rust_solver = VRPTWSolver(graph=graph,
                              instance=instance,
                              verbosity=1,
                              pricing_strategy="rust")
    rust_solver.solve()
    obj_colgen = rust_solver.rmp.getObjVal()
    obj_compact = solve_compact(instance, graph, 10).getObjVal()
    assert obj_colgen == obj_compact
