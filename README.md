# scip-routing
An exact VRPTW solver using [PySCIPOpt](https://github.com/scipopt/PySCIPOpt). Intended to provide an easier starting point for research in exact methods for vehicle routing problems. 


## Installation
The easiest way to get started is to create a new conda environment using the `env.yml` file. 

```bash 
conda env create -f env.yml
```

The code implements two pricers, i.e. solvers for the resource constrained shortest path subproblem, one in python and a faster one in Rust. To use the Rust one, use maturin to build the code in `rs_pricing`
```
maturin develop --release 
``` 


## Example
```python
import cvrplib

from scip_routing.solver import VRPTWSolver
from scip_routing.utils import instance_graph

instance, sol = cvrplib.download('R101', solution=True)

instance_graph = instance_graph(instance)

solver = VRPTWSolver(graph=instance_graph,
                     instance=instance,
                     verbosity=2,
                     pricing_strategy="rust", # "py" also can be used for the pure-python pricer
                     )
solver.solve()
```
