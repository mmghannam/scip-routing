import heapq
from collections import defaultdict

import networkx as nx
# from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
import pyscipopt as scip
from rs_pricing import Pricer as RustPricer


class Label:
    def __init__(self, last_node, cost, demand, earliest_time, last_label, visited=None):
        self.last_node = last_node
        self.cost = cost
        self.demand = demand
        self.earliest_time = earliest_time
        if visited is not None:
            self.visited = visited
        else:
            self.visited = set()
        self.last_label = last_label

    def __lt__(self, other):
        return self.earliest_time < other.earliest_time


class Pricer(scip.Pricer):
    """
    Solver for the Resource Constrained Shortest Path Problem, implements a basic Labeling Algorithm.
    """

    def __init__(self, graph, instance, init_added_paths={}, deleted_edges_from_node=set(), distance_fn=None,
                 strategy="py", verbosity=0):
        super().__init__()
        self.graph = graph
        self.instance = instance
        self.deleted_edges_from_node = deleted_edges_from_node
        self.start_depot = instance.depot
        self.end_depot = instance.n_customers + 1
        self.earliest = instance.earliest + [instance.earliest[self.start_depot]]
        self.latest = instance.latest + [instance.latest[self.start_depot]]
        self.demands = instance.demands + [instance.demands[self.start_depot]]
        self.service_times = instance.service_times + [instance.service_times[self.start_depot]]
        self.capacity = instance.capacity
        self.customers = instance.customers
        self.ncustomers = instance.n_customers
        self.verbosity = verbosity
        self.init_cons = None
        if distance_fn:
            self.distance_fn = distance_fn
        else:
            self.distance_fn = lambda i, j: self.graph[i][j]['distance']
        self.time_fn = lambda i, j: self.graph[i][j]['distance'] + self.service_times[i]
        self.added_paths = init_added_paths

        assert strategy in ["rust", "py"]
        self.strategy = strategy
        if strategy == "rust":
            self.rust_pricer = self.init_rust_pricer()
        
        self.elementary = False

    def init_rust_pricer(self) -> RustPricer:
        drive_times = self.instance.distances
        for drive_time_list in drive_times:
            drive_time_list.append(drive_time_list[0])

        neighbors = {n: list(self.graph.neighbors(n)) for n in self.customers}
        neighbors[self.start_depot] = self.customers
        neighbors[self.end_depot] = []

        time_windows = [(a, b) for a, b in zip(self.earliest, self.latest)]

        return RustPricer(
            demands=self.demands,
            time_windows=time_windows ,
            service_times=self.service_times,
            vehicle_capacity=self.capacity,
            customers=self.customers,
            start_depot=self.start_depot,
            end_depot=self.end_depot,
            drive_time=drive_times,
            neighbors=neighbors
        )

    def path_from_label(self, label: Label):
        curr = label
        path = []
        start_times = []
        cost = 0
        while curr is not None:
            path.insert(0, curr.last_node)
            start_times.insert(0, curr.earliest_time)
            if curr.last_label:
                cost += self.distance_fn(curr.last_label.last_node, curr.last_node)
            curr = curr.last_label
        return tuple(path), start_times, cost

    def add_path_to_rmp(self, path, cost, solver):
        customer_counts = {}
        for i in path:
            if i in customer_counts:
                customer_counts[i] += 1
            else:
                customer_counts[i] = 1

        row_ids = []
        row_coeffs = []
        for customer in solver.customers:
            if customer in path:
                row_ids.append(customer - 1)
                row_coeffs.append(customer_counts[customer])

        solver.rmp.addcols(objcoef=[cost],
                           start=[0, len(solver.customers)],
                           rowind=row_ids,
                           rowcoef=row_coeffs,
                           lb=[0],
                           ub=[float("inf")],
                           names=["-".join(map(str, path))],
                           types=['C'])

    def find_path_rust(self, duals, deleted_edges):
        return self.rust_pricer.find_path(duals, deleted_edges)

    def find_path(self, duals):
        deleted_edges = self.deleted_edges_from_node[self.model.getCurrentNode().getNumber()]
        if self.strategy == "py":
            result = list(self.find_path_py(duals, deleted_edges))
            result.sort(key=lambda element: str(element[0]))
            return result
        elif self.strategy == "rust":
            return self.find_path_rust(duals, deleted_edges)

    def find_path_py(self, duals, deleted_edges):
        unprocessed = {}
        processed = {}

        for i in range(self.ncustomers + 2):  # customers + end depot
            unprocessed[i] = set()
            processed[i] = set()

        start_label = Label(self.start_depot, 0, 0, 0, None)
        unprocessed[self.start_depot] = {start_label}
        label_queue = [(start_label.earliest_time, start_label.cost, start_label)]
        removed_labels = set()

        while label_queue:
            label_to_expand = self.choose_label_to_expand(label_queue, removed_labels)
            if not label_to_expand: break
            next_node_to_expand = label_to_expand.last_node

            for neighbor in nx.neighbors(self.graph, next_node_to_expand):
                if neighbor in label_to_expand.visited: continue
                if (next_node_to_expand, neighbor) in deleted_edges: continue
                demand, last_visited, redcost, earliest_time, visited = self.expand_label(duals, label_to_expand,
                                                                                          neighbor,
                                                                                          next_node_to_expand)

                new_label = Label(last_visited, redcost, demand, earliest_time, label_to_expand, visited)

                if self.is_feasible(demand, earliest_time, neighbor):
                    dominated = self.dominance_check({new_label}, processed[neighbor] | unprocessed[neighbor])
                    if len(dominated) == 0:
                        heapq.heappush(label_queue,
                                       (new_label.earliest_time, new_label.cost, new_label))
                        if neighbor != self.end_depot:
                            dominated = self.dominance_check(unprocessed[neighbor], {new_label})
                            removed_labels |= dominated
                            unprocessed[neighbor] = (unprocessed[neighbor] - dominated) | {new_label}
            processed[next_node_to_expand].add(label_to_expand)

        best_path_label = None
        best_path_redcost = float("inf")
        if processed[self.end_depot]:
            for label in processed[self.end_depot]:
                if label.cost < 1e-6:
                    yield *self.path_from_label(label), label.cost
                if label.cost < best_path_redcost:
                    best_path_label = label
                    best_path_redcost = label.cost
        best_path, start_times, best_path_travel_cost = self.path_from_label(best_path_label)
        yield best_path, start_times, best_path_travel_cost, best_path_redcost

    def choose_label_to_expand(self, label_heap, removed_labels):
        while label_heap:
            *_, label = heapq.heappop(label_heap)
            if label not in removed_labels:
                return label
            else:
                removed_labels.remove(label)  # this is done for the set size to not blow up
        return None

    def is_feasible(self, demand, earliest_time, neighbor):
        return demand <= self.capacity and earliest_time <= self.latest[neighbor]

    def expand_label(self, duals, label_to_expand, neighbor, next_node_to_expand):
        distance = self.distance_fn(next_node_to_expand, neighbor)
        redcost = label_to_expand.cost + (distance - duals[next_node_to_expand])
        last_visited = neighbor
        demand = label_to_expand.demand + self.demands[neighbor]
        earliest_time = max(label_to_expand.earliest_time + self.time_fn(next_node_to_expand, neighbor),
                            self.earliest[neighbor])
        visited = set(label_to_expand.visited)
        visited.add(neighbor)
        return demand, last_visited, redcost, earliest_time, visited

    def dominance_check(self, A, B):
        """
        Dominance check between two sets of labels

        :param A: set of labels
        :param B: set os labels
        :return: set of labels in A that are dominated by at least one label from B
        """
        dominated = set()
        for label_a in A:
            for label_b in B:
                if self.dominates(label_b, label_a):
                    dominated.add(label_a)
                    break
        return dominated

    def dominates(self, label_a, label_b):
        is_less_or_eq = label_a.cost <= label_b.cost and label_a.demand <= label_b.demand and \
                        label_a.earliest_time <= label_b.earliest_time
        one_is_strictly_less = label_a.cost < label_b.cost or label_a.demand < label_b.demand or \
                               label_a.earliest_time < label_b.earliest_time
        subset = self.elementary and label_a.visited.issubset(label_b.visited)
        return is_less_or_eq and one_is_strictly_less and subset

    def pricerredcost(self, *args, **kwargs):
        duals = {self.start_depot: 0}
        for i, c in enumerate(self.init_cons):
            duals[i + 1] = self.model.getDualsolLinear(c)
        duals[self.end_depot] = 0

        n_added_paths = 0
        min_redcost = 0

        done = False

        while not done:
            for path, start_times, cost, redcost in self.find_path(duals):
                path_name =str((*path,))
                if path_name not in self.added_paths:
                    if redcost < min_redcost:
                        min_redcost = redcost
                    n_added_paths += 1
                    if self.verbosity >= 3:
                        print(path, start_times, cost, redcost)

                    var = self.model.addVar(name=f"{str(path)}", obj=cost, vtype="B",
                                            pricedVar=True)
                    self.added_paths[path_name] = var
                    cust_i_in_path = defaultdict(lambda: 0)
                    for x in path:
                        cust_i_in_path[x] += 1
                    for i, cons in enumerate(self.init_cons):
                        if cust_i_in_path[i + 1] > 0:
                            # print(i + 1, cust_i_in_path[i + 1])
                            self.model.addConsCoeff(cons, var, cust_i_in_path[i + 1])
            if min_redcost == 0 and not self.get_elementary():
                self.set_elementary(True)
            else:
                done = True

        lowerbound = self.model.getLPObjVal() + min_redcost if self.get_elementary() else - float("inf")
        self.set_elementary(False)
        if self.verbosity >= 2:
            print(f"at{self.model.getCurrentNode().getNumber()}, LP obj:", self.model.getLPObjVal())
            if lowerbound < self.model.getCurrentNode().getLowerbound():
                print("lowerbound", self.model.getLPObjVal() + min_redcost)
        return {
            "result": scip.SCIP_RESULT.SUCCESS,
            "lowerbound": lowerbound
        }

    def set_elementary(self, val):
        if self.strategy == "py":
            self.elementary = val
        elif self.strategy == "rust":
            self.rust_pricer.set_elementary(val)

    def get_elementary(self):
        if self.strategy == "py":
            return self.elementary
        elif self.strategy == "rust":
            return self.rust_pricer.get_elementary()


    def pricerinit(self):
        for i, c in enumerate(self.init_cons):
            self.init_cons[i] = self.model.getTransformedCons(c)

    def set_init_cons(self, init_cons):
        self.init_cons = init_cons
