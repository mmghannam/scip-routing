use bit_set::BitSet;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{cmp::max, collections::BTreeMap, collections::BTreeSet, hash::Hash, sync::Arc};

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

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.partial_cmp(&other.id).unwrap()
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
        let mut processed = BTreeMap::<usize, BTreeSet<Arc<Label>>>::new();
        let mut unprocessed = BTreeMap::<usize, BTreeSet<Arc<Label>>>::new();

        let mut pred = BTreeMap::<usize, Arc<Label>>::new();

        for customer in self.customers.iter() {
            unprocessed.insert(*customer, BTreeSet::new());
            processed.insert(*customer, BTreeSet::new());
        }

        unprocessed.insert(self.start_depot, BTreeSet::new());
        unprocessed.insert(self.end_depot, BTreeSet::new());
        processed.insert(self.start_depot, BTreeSet::new());
        processed.insert(self.end_depot, BTreeSet::new());

        let mut current_label_id = 1;

        let start_label = Arc::new(Label::new(
            current_label_id,
            self.start_depot,
            0.0,
            0.0,
            0.0,
            self.time_windows[self.start_depot].0,
            BitSet::with_capacity(self.customers.len() + 2),
        ));

        current_label_id += 1;

        let mut label_queue = Vec::<Arc<Label>>::new();

        label_queue.push(start_label.clone());

        unprocessed.insert(self.start_depot, BTreeSet::from([start_label]));

        while let Some(label_to_expand) = label_queue.pop() {
            let next_node_to_expand = label_to_expand.last_node as usize;
            let neighbors = match self.neighbors.get(&next_node_to_expand) {
                Some(n) => n,
                None => continue,
            };

            let new_label_data: Vec<_> = neighbors
                .iter()
                .filter_map(|neighbor| {
                    current_label_id += 1;
                    self.process_label_expansion(
                        label_to_expand.clone(),
                        neighbor,
                        &duals,
                        &deleted_edges,
                        &processed,
                        &unprocessed,
                        &current_label_id,
                    )
                })
                .collect();

            // println!("new_label_data: {:?}", new_label_data);
            new_label_data.into_iter().for_each(|(new_label, dominated)| {
                label_queue.push(new_label.clone());
                pred.insert(new_label.id, label_to_expand.clone());
                let label_set_at_node = unprocessed.get_mut(&new_label.last_node).unwrap();
                if new_label.last_node != self.end_depot {
                    for label in dominated {
                        label_set_at_node.remove(&label);
                        pred.remove(&label.id);
                    }
                }
                label_set_at_node.insert(new_label.clone());
            });

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
    fn process_label_expansion(
        &self,
        label_to_expand: Arc<Label>,
        neighbor: &usize,
        duals: &BTreeMap<usize, f64>,
        deleted_edges: &BTreeSet<(usize, usize)>,
        processed: &BTreeMap<usize, BTreeSet<Arc<Label>>>,
        unprocessed: &BTreeMap<usize, BTreeSet<Arc<Label>>>,
        current_label_id: &usize,
    ) -> Option<(Arc<Label>, BTreeSet<Arc<Label>>)> {
        if label_to_expand.visited.contains(*neighbor) {
            return None;
        }
        if deleted_edges.contains(&(label_to_expand.last_node, *neighbor)) {
            return None;
        }

        let new_label =
            Arc::new(self.expand_label(&label_to_expand, *neighbor, &duals, current_label_id));

        if self.is_feasible(&new_label) {
            let label_set_at_node = unprocessed
                .get(neighbor)
                .unwrap()
                .iter()
                .chain(processed.get(neighbor).unwrap())
                .cloned()
                .collect::<BTreeSet<_>>();
            if !self.is_dominated(new_label.clone(), &label_set_at_node) {
                let dominated = self.dominated_by(new_label.clone(), &label_set_at_node);
                return Some((new_label, dominated));
            }
        }
        None
    }
    fn expand_label(
        &self,
        label_to_expand: &Label,
        neighbor: usize,
        duals: &BTreeMap<usize, f64>,
        current_label_id: &usize,
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
        a: &BTreeSet<Arc<Label>>,
        b: &BTreeSet<Arc<Label>>,
    ) -> BTreeSet<Arc<Label>> {
        // Which from A are dominated by a label in B
        let mut result = BTreeSet::<Arc<Label>>::new();
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

    fn is_dominated(&self, label: Arc<Label>, label_set: &BTreeSet<Arc<Label>>) -> bool {
        let set_for_label = BTreeSet::from([Arc::clone(&label)]);
        let dominated = self._dominance_check(&set_for_label, label_set);
        !dominated.is_empty()
    }

    fn dominated_by(
        &self,
        label: Arc<Label>,
        label_set: &BTreeSet<Arc<Label>>,
    ) -> BTreeSet<Arc<Label>> {
        self._dominance_check(label_set, &BTreeSet::from([Arc::clone(&label)]))
    }

    fn path_from_label(
        &self,
        label: &Label,
        pred: &BTreeMap<usize, Arc<Label>>,
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
