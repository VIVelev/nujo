from graphviz import Digraph

from nujo.autodiff._node import _Node
from nujo.autodiff.tensor import Tensor


class ComputationGraphPlotter:
    ''' Computation Graph Plotter

        Uses graphviz.

    '''
    def __init__(self, **kwargs):
        self.computation_graph = Digraph(**kwargs)

    @staticmethod
    def get_color(node: _Node) -> str:
        if isinstance(node, Tensor):
            if len(node.children) > 0:
                return 'lightblue'
            return 'indianred1'
        else:
            return 'gold2'

    @staticmethod
    def get_shape(node: _Node) -> str:
        if isinstance(node, Tensor):
            return 'box'
        else:
            return 'oval'

    def create(self,
               root: _Node,
               display_values=False) -> 'ComputationGraphPlotter':

        if len(root.children) == 0:
            return

        root_name = str(root) if display_values else repr(root)
        for child in root.children:
            child_name = str(child) if display_values else repr(child)

            self.computation_graph.node(child_name,
                                        color=self.get_color(child),
                                        shape=self.get_shape(child),
                                        style='filled')

            self.computation_graph.node(root_name,
                                        color=self.get_color(root),
                                        shape=self.get_shape(root),
                                        style='filled')

            self.computation_graph.edge(child_name, root_name)
            self.create(child)

        return self

    def view(self) -> None:
        self.computation_graph.view()
