import pyscipopt as scip
def solve(graph, instance, number_of_vehicles, verbosity=0): 
    model = scip.Model()
    
    if verbosity == 0:
        model.hideOutput()

    start_depot = instance.depot
    end_depot = instance.n_customers + 1
    earliest = instance.earliest + [instance.earliest[start_depot]]
    latest = instance.latest + [instance.latest[start_depot]]
    demands = instance.demands + [instance.demands[start_depot]]
    service_times = instance.service_times + [instance.service_times[start_depot]]
    
    # add variables
    vars = {}

    for k in range(number_of_vehicles):
        for i, j, data in graph.edges(data=True):
            var_name = str((i, j, k))
            var = model.addVar(obj=data["distance"], name=var_name, vtype="B")
            vars[i,j,k] = var
    start_vars = {}
    for k in range(number_of_vehicles):
        for i in [start_depot, end_depot] + instance.customers:
            var_name = str((i, k))
            var = model.addVar(obj=0, lb=0, ub=latest[i], name=var_name, vtype="C")
            start_vars[i,k] = var


    # set objective to minimize in pyscipopt
    model.setMinimize()

    ## add constraints
    # each customer is visited exactly once
    for customer in instance.customers:
        model.addCons(scip.quicksum(var for (i, _, _), var in vars.items() if i == customer) == 1)


    for k in range(number_of_vehicles):
        for node in graph.nodes:
            left_hand_side = scip.quicksum(vars[i, j, k] for i, j in graph.edges if i == node) - scip.quicksum(
                vars[i, j, k] for i, j in graph.edges if j == node)
            if node == start_depot:
                model.addCons(left_hand_side == 1)
            elif node == end_depot:
                model.addCons(left_hand_side == -1)
            else:
                model.addCons(left_hand_side == 0)

    for k in range(number_of_vehicles):
        model.addCons(scip.quicksum(demands[customer] * vars[i, j, k] for customer in instance.customers for i, j in graph.edges
                                   if i == customer) <= instance.capacity)
        
    for v in range(number_of_vehicles):
        for d in (start_depot, end_depot):
            model.addCons(earliest[d] <= start_vars[d, v])
            model.addCons(start_vars[d, v] <= latest[d])
    
    for k in range(number_of_vehicles):
        for customer in instance.customers:
            s = instance.earliest[customer]
            e = instance.latest[customer]
            outgoing_edge_vars = scip.quicksum(vars[i, j, k] for i, j in graph.edges if i == customer)
            model.addCons(s * outgoing_edge_vars <= start_vars[customer, k])
            model.addCons(start_vars[customer, k] <= e * outgoing_edge_vars)
    
    bigM = max(earliest[i] + service_times[i] + graph[i][j]["distance"] for i, j in graph.edges)
    for k in range(number_of_vehicles):
        for i, j in graph.edges:
            # nonlinear version
            # model.addCons(vars[i, j, k] * (start_vars[j, k] - service_times[i] - graph[i][j]["distance"] - start_vars[i, k]) >= 0)
            # linearized # todo: try out indicator constraints
            model.addCons(start_vars[i, k] +  service_times[i] + graph[i][j]["distance"] - bigM * (1 - vars[i, j, k]) <= start_vars[j, k])
    
    model.optimize()

    if verbosity > 0: 
        if model.getStatus() == "optimal":
            print("Optimal solution found")
            for k in range(number_of_vehicles):
                print(f"Vehicle {k}")
                for i, j in graph.edges:
                    if model.getVal(vars[i, j, k]) > 0.5:
                        print(f"Edge {i} -> {j}: obj: {vars[i, j, k].getObj()}")
                        print(f"Arrival time at {i} -> {model.getVal(start_vars[i, k])}")
                print(f"Start time: {model.getVal(start_vars[start_depot, k])}")
                print(f"End time: {model.getVal(start_vars[end_depot, k])}")

    return model
      