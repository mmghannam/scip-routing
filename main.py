import vrplib

from scip_routing.compact import solve_compact
from scip_routing.solver import VRPTWSolver
from scip_routing.utils import instance_graph, minify_instance

if __name__ == "__main__":
    vrplib.download_instance('R101', path='C101.vrp')
    instance = vrplib.read_instance('R101.vrp', instance_format="solomon")

    minify_instance(instance, 30)

    instance, graph = instance_graph(instance)
    # solve_compact(graph, instance, number_of_vehicles=5, verbosity=2)
    # py_solver = VRPTWSolver(graph=graph,
    #                         instance=instance,
    #                         verbosity=4,
    #                         pricing_strategy="py")
    # py_solver.solve()
    rust_solver = VRPTWSolver(graph=graph,
                              instance=instance,
                              verbosity=2,
                              pricing_strategy="rust")
    rust_solver.solve()

    # assert py_solver.rmp.getObjVal() == rust_solver.rmp.getObjVal() == 1031