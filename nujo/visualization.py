from graphviz import Digraph

from nujo.autodiff import Variable

__all__ = [
    'ComputationGraphPlotter',
]


class ComputationGraphPlotter:
    def __init__(self, **kwargs):
        self.computation_graph = Digraph(**kwargs)

    @staticmethod
    def get_color(node):
        if isinstance(node, Variable):
            if len(node.children) > 0:
                return 'lightblue'
            return 'indianred1'
        else:            
            return 'gray'

    @staticmethod
    def get_shape(node):
        if isinstance(node, Variable) and len(node.children) == 0:
            return 'box'
        else:
            return 'oval'
        
    def create(self, root):
        if len(root.children) == 0:
            return
        
        for child in root.children:
            self.computation_graph.node(
                repr(child),
                color=self.get_color(child),
                shape=self.get_shape(child),
                style='filled')
            
            self.computation_graph.node(
                repr(root),
                color=self.get_color(root),
                shape=self.get_shape(root),
                style='filled')
            
            self.computation_graph.edge(repr(child), repr(root))
            self.create(child)
            
    def view(self):
        self.computation_graph.view()
