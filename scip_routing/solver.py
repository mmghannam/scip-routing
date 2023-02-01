from collections import defaultdict

from scip_routing.edge_brancher import EdgeBrancher
from scip_routing.edge_branching_eventhdlr import EdgeBranchingEventhdlr
from scip_routing.pricing import Pricer
import pyscipopt as scip


class VRPTWSolver:
    def __init__(self, graph, instance, verbosity=0, distance_fn=None, pricing_strategy="rust"):
        self.start_depot = instance.depot
        self.end_depot = instance.n_customers + 1
        self.customers = instance.customers
        self.deleted_edges_from_node = defaultdict(lambda: set())
        self.graph = graph
        self.added_paths = {}
        self.pricer = Pricer(graph, instance, init_added_paths=self.added_paths,
                             deleted_edges_from_node=self.deleted_edges_from_node,
                             distance_fn=distance_fn,
                             strategy=pricing_strategy,
                             verbosity=verbosity)
        self.verbosity = verbosity
        self.rmp = self.init_rmp()
        init_cons = list(self.rmp.getConss())
        self.pricer.set_init_cons(init_cons)
        self.rmp.includePricer(self.pricer, "LabelingPricer", "")
        self.obj = float("inf")

    def init_rmp(self):
        rmp = scip.Model()
        for customer in self.customers:
            var_name = str((self.start_depot, customer, self.end_depot))
            cost = self.pricer.graph[self.start_depot][customer]["distance"] + \
                   self.pricer.graph[customer][self.end_depot]["distance"]
            var = rmp.addVar(obj=cost, name=var_name, vtype="B")
            self.added_paths[var_name] = var
            rmp.addCons(var == 1, separate=False, modifiable=True)
        rmp.setMinimize()
        if self.verbosity == 0:
            rmp.hideOutput()
        return rmp

    def solve(self):
        self.rmp.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        self.rmp.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.rmp.setSeparating(scip.SCIP_PARAMSETTING.OFF)
        self.rmp.disablePropagation()

        # include edge branching rule and its event handler
        branching_rule = EdgeBrancher(self.graph, self.deleted_edges_from_node)
        self.rmp.includeBranchrule(branching_rule, "Edge Branching Rule", "", priority=1000000, maxdepth=-1,
                                   maxbounddist=1)
        eventhdlr = EdgeBranchingEventhdlr(self.deleted_edges_from_node)
        self.rmp.includeEventhdlr(eventhdlr, "Edge Branching Event Handler", "")

        self.rmp.setParam("display/freq", 1)
        self.rmp.setParam("display/headerfreq", 1)
        self.rmp.setObjIntegral()
        self.rmp.optimize()
        if self.verbosity > 0:
            solution = self.rmp.getBestSol()
            print("Best solution found:")
            for var in self.rmp.getVars(transformed=True):
                solval = solution[var]
                if solval > 1e-6:
                    print(var, var.getObj(), solval)
