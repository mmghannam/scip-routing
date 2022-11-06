from collections import defaultdict

import networkx as nx
# from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
import pyscipopt as scip


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


class Pricer(scip.Pricer):
    """
    Solver for the Resource Constrained Shortest Path Problem, implements a basic Labeling Algorithm.
    """

    def __init__(self, graph, instance, distance_fn=None):
        super().__init__()
        self.graph = graph
        self.start_depot = instance.depot
        self.end_depot = instance.n_customers + 1
        self.earliest = instance.earliest + [instance.earliest[self.start_depot]]
        self.latest = instance.latest + [instance.latest[self.start_depot]]
        self.demands = instance.demands + [instance.demands[self.start_depot]]
        self.capacity = instance.capacity
        self.customers = instance.customers
        self.ncustomers = instance.n_customers
        self.init_cons = None
        if distance_fn:
            self.distance_fn = distance_fn
        else:
            self.distance_fn = lambda i, j: self.graph[i][j]['distance']
        self.time_fn = lambda i, j: self.graph[i][j]['distance']
        self.added_paths = {}
        self.elementary = False

    def path_from_label(self, label: Label):
        curr = label
        path = []
        cost = 0
        while curr is not None:
            path.insert(0, curr.last_node)
            if curr.last_label:
                cost += self.distance_fn(curr.last_label.last_node, curr.last_node)
            curr = curr.last_label
        return tuple(path), cost

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

    def find_path(self, duals):
        unprocessed = {}
        processed = {}

        for i in range(self.ncustomers + 2):  # customers + end depot
            unprocessed[i] = set()
            processed[i] = set()

        unprocessed[self.start_depot] = {Label(self.start_depot, 0, 0, 0, None)}
        unprocessed_count = 1

        while unprocessed_count > 0:
            label_to_expand = self.choose_label_to_expand(unprocessed)
            unprocessed_count -= 1
            # print('***', unprocessed_count)
            next_node_to_expand = label_to_expand.last_node
            for neighbor in nx.neighbors(self.graph, next_node_to_expand):
                if neighbor in label_to_expand.visited: continue
                demand, last_visited, redcost, earliest_time, visited = self.expand_label(duals, label_to_expand,
                                                                                          neighbor,
                                                                                          next_node_to_expand)

                new_label = Label(last_visited, redcost, demand, earliest_time, label_to_expand, visited)

                if self.is_feasible(demand, earliest_time, neighbor):
                    dominated = self.dominance_check({new_label}, processed[neighbor] | unprocessed[neighbor])
                    if len(dominated) == 0:
                        dominated = self.dominance_check(unprocessed[neighbor], {new_label})
                        unprocessed[neighbor] = (unprocessed[neighbor] - dominated) | {new_label}
                        unprocessed_count += 1 - len(dominated)
            processed[next_node_to_expand].add(label_to_expand)

        best_path_label = None
        best_path_redcost = float("inf")
        if processed[self.end_depot]:
            for label in processed[self.end_depot]:
                if label.cost < -1e-08:
                    yield *self.path_from_label(label), label.cost
                if label.cost < best_path_redcost:
                    best_path_label = label
                    best_path_redcost = label.cost
        best_path, best_path_travel_cost = self.path_from_label(best_path_label)
        yield best_path, best_path_travel_cost, best_path_redcost

    def is_feasible(self, demand, earliest_time, neighbor):
        return demand <= self.capacity and earliest_time <= self.latest[neighbor]

    def expand_label(self, duals, label_to_expand, neighbor, next_node_to_expand):
        distance = self.distance_fn(next_node_to_expand, neighbor)
        redcost = label_to_expand.cost + (distance - duals[next_node_to_expand])
        last_visited = neighbor
        demand = label_to_expand.demand + self.demands[neighbor]
        earliest_time = max(label_to_expand.earliest_time + self.time_fn(next_node_to_expand, neighbor),
                            self.latest[neighbor])
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

    @staticmethod
    def choose_label_to_expand(unprocessed):
        # todo: choose one with min time
        # returns first label it sees
        for customer, labels in unprocessed.items():
            if len(labels):
                return labels.pop()
        return None

    def dominates(self, label_a, label_b):
        is_less_or_eq = label_a.cost <= label_b.cost and label_a.demand <= label_b.demand and \
                        label_a.earliest_time <= label_b.earliest_time
        one_is_strictly_less = label_a.cost < label_b.cost or label_a.demand < label_b.demand or \
                               label_a.earliest_time < label_b.earliest_time
        if self.elementary:
            visited_less = label_a.visited.issubset(label_b.visited)
        else:
            visited_less = True
        return is_less_or_eq and one_is_strictly_less and visited_less

    def pricerredcost(self, *args, **kwargs):
        duals = {self.start_depot: 0}
        for i, c in enumerate(self.init_cons):
            duals[i + 1] = self.model.getDualsolLinear(c)
        duals[self.end_depot] = 0

        n_added_paths = 0
        self.elementary = True

        while True:
            something_added = False
            for path, cost, redcost in self.find_path(duals):
                if redcost < -1e-8 and str(path) not in self.added_paths:
                    something_added = True
                    n_added_paths += 1
                    # print(path, redcost)

                    var = self.model.addVar(name=f"var-{str(path)}", obj=cost, vtype="C",
                                            pricedVar=True)
                    self.added_paths[str(path)] = var
                    cust_i_in_path = defaultdict(lambda: 0)
                    for x in path:
                        cust_i_in_path[x] += 1
                    for i, cons in enumerate(self.init_cons):
                        if cust_i_in_path[i + 1] > 0:
                            # print(i + 1, cust_i_in_path[i + 1])
                            self.model.addConsCoeff(cons, var, cust_i_in_path[i + 1])
            if not something_added and not self.elementary:
                print("switched to elementary mode")
                self.elementary = True
                self.first_found = False
            else:
                break

        return {"result": scip.SCIP_RESULT.SUCCESS}

    def pricerinit(self):
        for i, c in enumerate(self.init_cons):
            self.init_cons[i] = self.model.getTransformedCons(c)

    def set_init_cons(self, init_cons):
        self.init_cons = init_cons
