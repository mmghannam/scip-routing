use bit_set::BitSet;
use std::{cmp::max, collections::HashMap, collections::HashSet, hash::Hash, rc::Rc};
use std::collections::BinaryHeap;

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
    visited: BitSet,
}

impl Label {
    fn new(
        id: usize,
        last_node: usize,
        cost: f64,
        reduced_cost: f64,
        demand: f64,
        earliest_time: usize,
        visited: BitSet,
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

impl Ord for Label {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.earliest_time
            .cmp(&other.earliest_time).reverse()
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[pyclass]
#[derive(Debug)]
struct Pricer {
    demands: Vec<usize>,
    time_windows: Vec<(usize, usize)>,
    service_times: Vec<usize>,
    vehicle_capacity: usize,
    customers: Vec<usize>,
    start_depot: usize,
    end_depot: usize,
    drive_time: Vec<Vec<usize>>,
    neighbors: BTreeMap<usize, Vec<usize>>,
    elementary: bool,
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
        neighbors: BTreeMap<usize, Vec<usize>>,
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
            elementary: false,
        }
    }

    fn get_elementary(&self) -> PyResult<bool> {
        Ok(self.elementary)
    }

    fn set_elementary(&mut self, value: bool) -> PyResult<()> {
        self.elementary = value;
        Ok(())
    }

    fn find_path(
        &self,
        duals: BTreeMap<usize, f64>,
        deleted_edges: BTreeSet<(usize, usize)>,
    ) -> Vec<(Vec<usize>, Vec<usize>, f64, f64)> {
        let mut processed = BTreeMap::<usize, BTreeSet<Rc<Label>>>::new();
        let mut unprocessed = BTreeMap::<usize, BTreeSet<Rc<Label>>>::new();

        let mut pred = BTreeMap::<usize, Rc<Label>>::new();

        for customer in self.customers.iter() {
            unprocessed.insert(*customer, BTreeSet::new());
            processed.insert(*customer, BTreeSet::new());
        }

        unprocessed.insert(self.start_depot, BTreeSet::new());
        unprocessed.insert(self.end_depot, BTreeSet::new());
        processed.insert(self.start_depot, BTreeSet::new());
        processed.insert(self.end_depot, BTreeSet::new());

        let mut current_label_id = 1;

        let start_label = Rc::new(Label::new(
            current_label_id,
            self.start_depot,
            0.0,
            0.0,
            0.0,
            self.time_windows[self.start_depot].0,
            BitSet::with_capacity(self.customers.len() + 2),
        ));

        current_label_id += 1;

        // let mut label_queue = Vec::<Rc<Label>>::new();
        let mut label_queue = BinaryHeap::<Rc<Label>>::new();

        label_queue.push(start_label.clone());

        unprocessed.insert(self.start_depot, BTreeSet::from([start_label]));

        while let Some(label_to_expand) = label_queue.pop() {
            let next_node_to_expand = label_to_expand.last_node as usize;
            let neighbors = match self.neighbors.get(&next_node_to_expand) {
                Some(n) => n,
                None => continue,
            };

            for neighbor in neighbors {
                if label_to_expand.visited.contains(*neighbor) {
                    continue;
                }
                if deleted_edges.contains(&(label_to_expand.last_node, *neighbor)) {
                    continue;
                }

                let new_label = Rc::new(self.expand_label(
                    &label_to_expand,
                    *neighbor,
                    &duals,
                    &mut current_label_id,
                ));

                if self.is_feasible(&new_label) {
                    let label_set_at_node = unprocessed.get_mut(neighbor).unwrap();
                    let label_set_at_node_processed = processed.get_mut(neighbor).unwrap();
                    if !self.is_dominated(new_label.clone(), &label_set_at_node) && !self.is_dominated(new_label.clone(), &label_set_at_node_processed) {
                        label_queue.push(new_label.clone());
                        pred.insert(new_label.id, label_to_expand.clone());
                        if neighbor != &self.end_depot {
                            let dominated =
                                self.dominated_by(new_label.clone(), &label_set_at_node);
                            for label in dominated {
                                label_set_at_node.remove(&label);
                                pred.remove(&label.id);
                            }
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

        let empty_set = BTreeSet::new();
        let mut redcost_paths = vec![] as Vec<(Vec<usize>, Vec<usize>, f64, f64)>;
        let labels_at_end_depot = match processed.get(&self.end_depot) {
            Some(l) => Box::new(l),
            None => Box::new(&empty_set),
        };

        for label in labels_at_end_depot.iter() {
            let cost = label.reduced_cost;
            if cost < 1e-6 {
                let (path, start_times) = self.path_from_label(label, &pred);
                redcost_paths.push((path, start_times, label.cost, label.reduced_cost));
            }
        }
        // redcost_paths.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        redcost_paths
    }
}

// Methods visible only to rust
impl Pricer {
    fn expand_label(
        &self,
        label_to_expand: &Label,
        neighbor: usize,
        duals: &BTreeMap<usize, f64>,
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
        let reduced_cost = label_to_expand.reduced_cost + (distance as f64 - duals[&last_node]);

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

    fn dominates(&self, la: &Label, lb: &Label) -> bool {
        let less_then_or_eq = la.earliest_time <= lb.earliest_time
            && la.reduced_cost <= lb.reduced_cost
            && la.demand <= lb.demand;
        let one_is_less = la.earliest_time < lb.earliest_time
            || la.reduced_cost < lb.reduced_cost
            || la.demand < lb.demand;
        let dominates_non_elementary = less_then_or_eq && one_is_less;
        if self.elementary {
            dominates_non_elementary && la.visited.is_subset(&lb.visited)
        } else {
            dominates_non_elementary
        }
    }

    fn _dominance_check(
        &self,
        a: &BTreeSet<Rc<Label>>,
        b: &BTreeSet<Rc<Label>>,
    ) -> BTreeSet<Rc<Label>> {
        // Which from A are dominated by a label in B
        let mut result = BTreeSet::<Rc<Label>>::new();
        for la in a {
            for lb in b {
                if self.dominates(lb, la) {
                    result.insert(la.clone());
                    break;
                }
            }
        }
        result
    }

    fn is_dominated(&self, label: Rc<Label>, label_set: &BTreeSet<Rc<Label>>) -> bool {
        let set_for_label = BTreeSet::from([Rc::clone(&label)]);
        let dominated = self._dominance_check(&set_for_label, label_set);
        !dominated.is_empty()
    }

    fn dominated_by(
        &self,
        label: Rc<Label>,
        label_set: &BTreeSet<Rc<Label>>,
    ) -> BTreeSet<Rc<Label>> {
        self._dominance_check(label_set, &BTreeSet::from([Rc::clone(&label)]))
    }

    fn path_from_label(
        &self,
        label: &Label,
        pred: &BTreeMap<usize, Rc<Label>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut path = Vec::<usize>::new();
        let mut start_times = Vec::<usize>::new();
        let mut current_label = label;
        while let Some(parent) = pred.get(&current_label.id) {
            path.push(current_label.last_node);
            start_times.push(current_label.earliest_time);
            current_label = parent;
        }
        path.push(current_label.last_node);
        start_times.push(current_label.earliest_time);
        path.reverse();
        start_times.reverse();
        (path, start_times)
    }
}
