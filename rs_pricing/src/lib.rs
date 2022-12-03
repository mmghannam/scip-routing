use std::{cmp::max, collections::HashMap, collections::HashSet, hash::Hash};

use pyo3::prelude::*;

#[pymodule]
fn rs_pricing(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Pricer>()?;
    Ok(())
}

// class Label:
//     def __init__(self, last_node, cost, demand, earliest_time, last_label, visited=None):
//         self.last_node = last_node
//         self.cost = cost
//         self.demand = demand
//         self.earliest_time = earliest_time
//         if visited is not None:
//             self.visited = visited
//         else:
//             self.visited = set()
//         self.last_label = last_label

//     def __lt__(self, other):
//         return self.earliest_time < other.earliest_time

#[derive(Debug)]
struct Label<'a> {
    id: usize,
    last_node: usize,
    cost: f64,
    reduced_cost: f64,
    demand: f64,
    earliest_time: usize,
    visited: HashSet<usize>,
    last_label: Option<Box<&'a Label<'a>>>,
}

impl<'a> Label<'a> {
    fn new(
        id: usize,
        last_node: usize,
        cost: f64,
        reduced_cost: f64,
        demand: f64,
        earliest_time: usize,
        visited: HashSet<usize>,
        last_label: Option<Box<&Label<'a>>>,
    ) -> Self {
        Self {
            id,
            last_node,
            cost,
            reduced_cost,
            demand,
            earliest_time,
            visited,
            last_label,
        }
    }
}

impl PartialEq for Label<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Label<'_> {}

impl Hash for Label<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[pyclass]
struct Pricer {
    demands: Vec<usize>,
    time_windows: Vec<(usize, usize)>,
    service_times: Vec<usize>,
    vehicle_capacity: usize,
    customers: Vec<usize>,
    start_depot: usize,
    end_depot: usize,
    drive_time: Vec<Vec<usize>>,
    neighbors: HashMap<usize, Vec<usize>>,
}

// methods for Pricer to export to python
#[pymethods]
impl Pricer {
    #[new]
    fn new(
        demands: Vec<usize>,
        time_windows: Vec<(usize, usize)>,
        service_times: Vec<usize>,
        vehicle_capacity: usize,
        customers: Vec<usize>,
        start_depot: usize,
        end_depot: usize,
        drive_time: Vec<Vec<usize>>,
        neighbors: HashMap<usize, Vec<usize>>,
    ) -> Self {
        Self {
            demands,
            time_windows,
            service_times,
            vehicle_capacity,
            customers,
            start_depot,
            end_depot,
            drive_time,
            neighbors,
        }
    }

    fn find_path(
        &self,
        duals: HashMap<usize, f64>,
        deleted_edges: HashSet<(usize, usize)>,
    ) -> (Vec<usize>, f64, f64) {
        let mut processed = HashMap::<usize, HashSet<&Label>>::new();
        let mut unprocessed = HashMap::<usize, HashSet<&Label>>::new();

        for customer in self.customers.iter() {
            unprocessed.insert(*customer, HashSet::new());
            processed.insert(*customer, HashSet::new());
        }

        let mut current_label_id = 1;

        let start_label = Label::new(
            current_label_id,
            self.start_depot,
            0.0,
            0.0,
            0.0,
            self.time_windows[self.start_depot].0,
            HashSet::new(),
            None,
        );

        current_label_id += 1;

        let mut label_queue = Vec::<&Label>::new();

        label_queue.push(&start_label);
        unprocessed
            .get_mut(&self.start_depot)
            .unwrap()
            .insert(&start_label);

        while let Some(label_to_expand) = label_queue.pop() {
            let next_node_to_expand = label_to_expand.last_node as usize;
            let neighbors = self.neighbors.get(&next_node_to_expand).unwrap();
            for neighbor in neighbors {
                if label_to_expand.visited.contains(neighbor) {
                    continue;
                }
                if deleted_edges.contains(&(label_to_expand.last_node, *neighbor)) {
                    continue;
                }

                let new_label =
                    self.expand_label(label_to_expand, *neighbor, &duals, &mut current_label_id);

                if self.is_feasible(&new_label) {
                    let label_set_at_node = unprocessed.get_mut(neighbor).unwrap();
                    if !self.is_dominated(&new_label, label_set_at_node) {
                        label_queue.push(&new_label);
                        let dominated = self.dominated_by(&new_label, label_set_at_node);
                        for label in dominated {
                            label_set_at_node.remove(label);
                        }
                        label_set_at_node.insert(&new_label);
                    }
                }
            }
            processed
                .get_mut(&next_node_to_expand)
                .unwrap()
                .insert(label_to_expand);
        }

        let mut best_label: Option<&Label> = None;
        let mut best_cost = f64::INFINITY;
        let labels_at_end_depot = processed.get(&self.end_depot).unwrap();
        for label in labels_at_end_depot {
            let cost = label.reduced_cost;
            if cost < best_cost {
                best_cost = cost;
                best_label = Some(label);
            }
        }

        match best_label {
            Some(label) => {
                let best_label = best_label.unwrap();
                (
                    self.path_from_label(best_label),
                    best_label.reduced_cost,
                    best_label.cost,
                )
            }
            None => {
                panic!("No feasible path found");
            }
        }
    }
}

// Methods visible only to rust
impl Pricer {
    fn expand_label(
        &self,
        label_to_expand: &Label,
        neighbor: usize,
        duals: &HashMap<usize, f64>,
        current_label_id: &mut usize,
    ) -> Label {
        let distance = self.drive_time[label_to_expand.last_node][neighbor];
        let last_node = label_to_expand.last_node;

        let next_earliest_time = max(
            label_to_expand.earliest_time
                + self.service_times[label_to_expand.last_node]
                + distance,
            self.time_windows[neighbor].0,
        );

        let cost = label_to_expand.cost + distance as f64;
        let reduced_cost = label_to_expand.reduced_cost + distance as f64 - duals[&last_node];

        let accumulated_demand = label_to_expand.demand + self.demands[neighbor] as f64;

        let mut visited = label_to_expand.visited.clone();
        visited.insert(neighbor);

        let new_label = Label::new(
            *current_label_id,
            neighbor,
            cost,
            reduced_cost,
            accumulated_demand,
            next_earliest_time,
            visited,
            Some(Box::new(label_to_expand.clone())),
        );

        *current_label_id += 1;

        new_label
    }

    fn is_feasible(&self, label: &Label) -> bool {
        label.earliest_time <= self.time_windows[label.last_node].1
            && label.demand <= self.vehicle_capacity as f64
    }

    fn dominates(la: &Label, lb: &Label) -> bool {
        let less_then_or_eq = la.earliest_time <= lb.earliest_time
            && la.reduced_cost <= lb.reduced_cost
            && la.demand <= lb.demand;
        let one_is_less = la.earliest_time < lb.earliest_time
            || la.reduced_cost < lb.reduced_cost
            || la.demand < lb.demand;
        less_then_or_eq && one_is_less
    }

    fn _dominance_check(&self, A: &HashSet<&Label>, B: &HashSet<&Label>) -> HashSet<&Label> {
        // Which from A are dominated by a label in B
        let mut result = HashSet::<&Label>::new();
        for a in A {
            for b in B {
                if Self::dominates(b, a) {
                    result.insert(a);
                    break;
                }
            }
        }
        result
    }

    fn is_dominated(&self, label: &Label, label_set: &HashSet<&Label>) -> bool {
        let dominated = self._dominance_check(&HashSet::from([label]), label_set);
        !dominated.is_empty()
    }

    fn dominated_by(&self, label: &Label, label_set: &HashSet<&Label>) -> HashSet<&Label> {
        self._dominance_check(label_set, &HashSet::from([label]))
    }

    fn path_from_label(&self, label: &Label) -> Vec<usize> {
        let mut path = Vec::<usize>::new();
        let mut current_label = Some(Box::new(label));
        while let Some(label) = current_label {
            path.push(label.last_node);
            current_label = label.last_label.clone();
        }
        path.reverse();
        path
    }
}
