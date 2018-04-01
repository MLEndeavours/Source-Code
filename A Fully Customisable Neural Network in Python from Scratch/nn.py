import numpy as np
np.random.seed(777) # You can put any int here.

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
        
    def predict(self, x, to_predict=True, argmax=True, rand_weights=False):
        """
        Performs a pass of forward propagation.
        If to_predict is set to True, trained weights are used
        and predictions are returned in a single vector 
        with labels from 0 to (n_classes - 1).
        """
        
        nodes = self.nodes
        layers = {}
        weights = {}

        # -------------- for input layer
        m = x.shape[0]
        x = np.append(np.ones(m).reshape(m, 1), x, axis=1)
        layers['a%d' % 1] = x

        # --------------- for dense layers
        for i in range(len(nodes)): 
            m, n = x.shape
            w = np.random.randn(nodes[i], n) if rand_weights else self.weights['w%d' % (i + 1)]
            z = x.dot(w.T)
            a = np.append(np.ones(m).reshape(m, 1), self.sigmoid(z), axis=1)

            if not to_predict:
                layers['a%d' % (i + 2)] = a # We start from a2, as a1 is already done. 
                weights['w%d' % (i + 1)] = w # We start from w1
            x = a # to repeat the same procedure for the next layer.

        # --------------- for output layer
        m, n = x.shape
        w = np.random.randn(self.n_classes, n) if rand_weights else self.weights['w%d' % (len(nodes) + 1)]
        z = x.dot(w.T)
        a = self.sigmoid(z)
        output = a

        if not to_predict:
            layers['a%d' % (len(layers) + 1)] = a
            weights['w%d' % (len(weights) + 1)] = w

        if to_predict:
            return np.argmax(output, axis=1)
        elif rand_weights:
            self.layers = layers
            self.weights = weights
        else:
            return layers, weights
        
