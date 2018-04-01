import numpy as np
class NN(object):
    """
    A network that uses sigmoid activation function.
    """
    
    def __init__(self):

        self.nodes = []
        self.layers = {}
        self.weights = {}
        self.n_classes = 0

    def add_layer(self, n_nodes, output_layer=False):
        """
        Adds a layer of specified no. of output nodes.  
        For the output layer, the flag  output_layer must be True.
        A network must have an output layer.
        """

        if not output_layer:
            self.nodes.append(n_nodes)
        else:
            self.n_classes = n_nodes
 
     def sigmoid(self, z):
        """
        Calculates the sigmoid activation function.
        """

        return 1 / (1 + np.exp(-z))
        
