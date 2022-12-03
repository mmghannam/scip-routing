use std::{cmp::max, collections::HashMap, collections::HashSet, hash::Hash};

use pyo3::prelude::*;

#[pymodule]
fn rs_pricing(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Pricer>()?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Label {
    id: usize,
    last_node: usize,
    cost: f64,
    reduced_cost: f64,
    demand: f64,
    earliest_time: usize,
    visited: HashSet<usize>,
}

impl Label {
    fn new(
        id: usize,
        last_node: usize,
        cost: f64,
        reduced_cost: f64,
        demand: f64,
        earliest_time: usize,
        visited: HashSet<usize>,
    ) -> Self {
        Self {
            id,
            last_node,
            cost,
            reduced_cost,
            demand,
            earliest_time,
            visited,
        }
    }
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Label {}

impl Hash for Label {
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
        let mut processed = HashMap::<usize, HashSet<Label>>::new();
        let mut unprocessed = HashMap::<usize, HashSet<Label>>::new();

        let mut pred = HashMap::<usize, Label>::new();

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
        );

        current_label_id += 1;

        let mut label_queue = Vec::<Label>::new();

        label_queue.push(start_label.clone());
        
        unprocessed
            .get_mut(&self.start_depot)
            .unwrap()
            .insert(start_label.clone());

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
                    self.expand_label(&label_to_expand, *neighbor, &duals, &mut current_label_id);

                if self.is_feasible(&new_label) {
                    let label_set_at_node = unprocessed.get_mut(neighbor).unwrap();
                    if !self.is_dominated(new_label.clone(), label_set_at_node) {
                        label_queue.push(new_label.clone());
                        pred.insert(new_label.id, label_to_expand.clone());
                        let dominated = self.dominated_by(new_label.clone(), &label_set_at_node);
                        for label in dominated {
                            label_set_at_node.remove(&label);
                        }
                        label_set_at_node.insert(new_label.clone());
                    }
                }
            }
            processed
                .get_mut(&next_node_to_expand)
                .unwrap()
                .insert(label_to_expand.clone());
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
                (
                    self.path_from_label(label, &pred),
                    label.reduced_cost,
                    label.cost,
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

    fn _dominance_check(&self, A: &HashSet<Label>, B: &HashSet<Label>) -> HashSet<Label> {
        // Which from A are dominated by a label in B
        let mut result = HashSet::<Label>::new();
        for a in A {
            for b in B {
                if Self::dominates(b, a) {
                    result.insert(a.clone());
                    break;
                }
            }
        }
        result
    }

    fn is_dominated(&self, label: Label, label_set: &HashSet<Label>) -> bool {
        let set_for_label = HashSet::from([label]); 
        let dominated = self._dominance_check(&set_for_label, label_set);
        !dominated.is_empty()
    }

    fn dominated_by(&self, label: Label, label_set: &HashSet<Label>) -> HashSet<Label> {
        self._dominance_check(label_set, &HashSet::from([label]))
    }

    fn path_from_label(&self, label: &Label, pred: &HashMap<usize, Label>) -> Vec<usize> {
        let mut path = Vec::<usize>::new();
        let mut current_label = label;
        while let Some(parent) = pred.get(&current_label.id) {
            path.push(current_label.last_node);
            current_label = parent;
        }
        path.reverse();
        path
    }
}