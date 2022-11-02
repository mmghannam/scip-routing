from time import time

from scip_routing.pricing import Pricer
import pyscipopt as scip


class VRPTWSolver:
    def __init__(self, graph, instance, verbose=False, distance_fn=None):
        self.start_depot = instance.depot
        self.end_depot = instance.n_customers + 1
        self.customers = instance.customers
        self.pricer = Pricer(graph, instance, distance_fn=distance_fn)
        self.rmp = self.init_rmp()
        init_cons = list(self.rmp.getConss())
        self.pricer.set_init_cons(init_cons)
        self.rmp.includePricer(self.pricer, "LabelingPricer", "")
        self.verbose = verbose
        self.obj = float("inf")

    def init_rmp(self):
        rmp = scip.Model()
        vars = []
        for customer in self.customers:
            var_name = f"{self.start_depot}-{customer}-{self.end_depot}"
            cost = self.pricer.graph[self.start_depot][customer]["distance"] + \
                   self.pricer.graph[customer][self.end_depot]["distance"]
            var = rmp.addVar(obj=cost, name=var_name)
            vars.append(var)

            rmp.addCons(var == 1, separate=False, modifiable=True)
        rmp.setMinimize()
        return rmp

    def solve(self):
        self.rmp.optimize()
