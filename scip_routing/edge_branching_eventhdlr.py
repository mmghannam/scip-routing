import pyscipopt as scip

from scip_routing.utils import var_to_edges


class EdgeBranchingEventhdlr(scip.Eventhdlr):
    def __init__(self, deleted_edges_from_node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deleted_edges_from_node = deleted_edges_from_node

    def eventinit(self):
        self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        deleted_edges = self.deleted_edges_from_node[self.model.getCurrentNode().getNumber()]
        for var in self.model.getVars(transformed=True):
            var_edges = var_to_edges(var)
            if any(edge in var_edges for edge in deleted_edges):
                self.model.chgVarUb(var, 0)