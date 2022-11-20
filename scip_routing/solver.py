from collections import defaultdict

from scip_routing.edge_brancher import EdgeBrancher
from scip_routing.edge_branching_eventhdlr import EdgeBranchingEventhdlr
from scip_routing.pricing import Pricer
import pyscipopt as scip


class VRPTWSolver:
    def __init__(self, graph, instance, verbose=False, distance_fn=None):
        self.start_depot = instance.depot
        self.end_depot = instance.n_customers + 1
        self.customers = instance.customers
        self.deleted_edges_from_node = defaultdict(lambda: set())
        self.graph = graph
        self.pricer = Pricer(graph, instance, deleted_edges_from_node=self.deleted_edges_from_node, distance_fn=distance_fn)
        self.verbose = verbose
        self.rmp = self.init_rmp()
        init_cons = list(self.rmp.getConss())
        self.pricer.set_init_cons(init_cons)
        self.rmp.includePricer(self.pricer, "LabelingPricer", "")
        self.obj = float("inf")

    def init_rmp(self):
        rmp = scip.Model()
        for customer in self.customers:
            var_name = f"{self.start_depot}-{customer}-{self.end_depot}"
            cost = self.pricer.graph[self.start_depot][customer]["distance"] + \
                   self.pricer.graph[customer][self.end_depot]["distance"]
            var = rmp.addVar(obj=cost, name=var_name, vtype="B")
            rmp.addCons(var == 1, separate=False, modifiable=True)
        rmp.setMinimize()
        if not self.verbose:
            rmp.hideOutput()
        return rmp

    def solve(self):
        self.rmp.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        self.rmp.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.rmp.disablePropagation()

        # include edge branching rule and its event handler
        branching_rule = EdgeBrancher(self.graph, self.deleted_edges_from_node)
        self.rmp.includeBranchrule(branching_rule, "Edge Branching Rule", "", priority=1000000, maxdepth=-1,
                                   maxbounddist=1)
        eventhdlr = EdgeBranchingEventhdlr(self.deleted_edges_from_node)
        self.rmp.includeEventhdlr(eventhdlr, "Edge Branching Event Handler", "")


        self.rmp.optimize()
        # print(self.rmp.getBestSol())
