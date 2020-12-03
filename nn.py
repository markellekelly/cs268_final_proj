import numpy as np

class NeuralNetwork:
    
    """
    A class representing a fully-connected, feed-forward neural network.
    
    Params:
        nlayers: the total number of hidden layers in the network
        nnodes: an array containing the number of nodes in each layer
        activations: an array containing the names of activation functions to use in each layer
            (note that activations[0] and activations[nlayers+2] will not be used)
            
    Notes:
        All hidden layers and the input layer also include an intercept node, not counted
        in nnodes. Intercept nodes are not connected to any nodes in the previous layer
        and have a value of 1; they are connected to all nodes in the next layer.
    """
    
    def __init__(self, nlayers, nnodes, activations):
        assert nlayers == len(nnodes) - 2
        assert nlayers == len(activations) - 2
        self.nlayers = nlayers
        self.nnodes = nnodes
        self.activations = activations
        self.weights = self.initialize_weights()
        self.z = []
        self.h = []
        
    def initialize_weights(self):
        """ 
        Randomly initialize all weights to numbers in [-0.25,0.25].
        
        Returns:
            An array of length nlayers+1 where element i is an array of length 
            (nnodes[i]+1)*nnodes[i+1] containing random weights for each edge between
            layers i and i+1, including weights corresponding to the intercept node.
        """
        weights = []
        for i in range(self.nlayers+1):
            weights_i = []
            for j in range((self.nnodes[i]+1)*self.nnodes[i+1]):
                weights_i.append(np.random.uniform(-0.25,0.25))
            weights.append(weights_i)
        return weights
    
    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def forward_prop(self, input_data):
        """ 
        Propagates input matrix x through the neural network, saving intermediate values
        of z and h. Adds intercept node to each layer.
        
        Params:
            inputs: an array of arrays of input values
        
        Returns:
            an array of output values
        """
        assert len(input_data[0]) == self.nnodes[0]
        self.z=[]
        inputs=[0]
        z=[]
        inputs[0] = np.repeat(1,len(input_data))
        for i in range(len(input_data[0])):
            add = np.array([j[i] for j in input_data])
            inputs.append(add)
            z.append(add)
        self.z.append(z)
        self.h = self.z.copy()
        for i in range(self.nlayers+1):
            new_nodes = [0] * self.nnodes[i+1]
            p = 0
            #go through the weights for each input node one group at a time 
            for w in self.chunker(self.weights[i], len(new_nodes)):
                for j in range(len(w)):
                    # add the value corresponding to the jth node in the next layer
                    new_nodes[j]= new_nodes[j]+ w[j]*inputs[p]
                p += 1
            self.z.append(new_nodes)
            self.h.append([self.activations[i+1](j) for j in new_nodes])
            inputs = self.h[i+1].copy()
            inputs.insert(0,np.repeat(1,len(inputs[0])))
        return self.z[self.nlayers+1]
    
    def back_prop(self, y_pred, y, rate, derivs):
        """ 
        Performs one iteration of backpropagation.
        
        Params:
            y_pred: the predicted y value
            y: the true y value
            rate: the learning rate
            derivs: array of the derivatives of each activation function
                (note again that derivs[0] and derivs[nlayers+2] will not be used))
        
        Returns:
            an array of updated weights
        """
        deltas = y_pred - y[0]
        new_weights = []
        for layer in range(self.nlayers,-1,-1):
            i=0
            new_w = []
            n = len(self.h[layer][0])
            h_vals = self.h[layer].copy()
            h_vals.insert(0,np.repeat(1,n))
            for h in h_vals:
                for d in deltas:
                    #sum weight changes across observations
                    changes = np.sum(h*d)
                    old_w = self.weights[layer][i]
                    new_w.append(old_w - rate*changes)
                    i+=1
            new_weights.insert(0,new_w)
            i=len(self.z[layer+1])
            new_deltas = []
            for z in self.z[layer]:
                new_d=0
                for d in deltas:
                    new_d += derivs[layer](z)*d*self.weights[layer][i]
                    i+=1
                new_deltas.append(new_d)
            deltas = new_deltas
        self.weights = new_weights
                
    def gradient_descent(self, data, y_val, rate, batch_size, derivs, tol):
        """ 
        Performs stochastic gradient descent to train the weights of the neural network.
        
        Params:
            data: the full dataset, not including the target variable
            y_val: the target column
            rate: the learning rate
            batch_size: the number of observations in each batch
            derivs: array of the derivatives of each activation function
                (note again that derivs[0] and derivs[nlayers+2] will not be used))
            tol: tolerance (difference in MSEs to stop at)
        
        Returns:
            an array of weights of optimal neural network
        """
        diff = 100
        new_MSE = 0
        while diff > tol:
            old_MSE = new_MSE
            new_MSE = 0
            prev = 0
            while prev < len(data):
                nxt = prev+batch_size
                if nxt > len(data):
                    nxt = len(data)
                xs = data.values[prev:nxt]
                ys = y_val.values[prev:nxt]
                prev = nxt
                y_pred = self.forward_prop(xs)
                self.back_prop(y_pred,ys,rate,derivs)
                new_MSE += np.sum((self.forward_prop(xs)[0]-ys[0])**2)
            new_MSE = new_MSE/len(data)
            diff = abs(old_MSE - new_MSE)
            #print("MSE = "+str(new_MSE))
        return new_MSE