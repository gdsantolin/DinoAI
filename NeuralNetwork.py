import numpy as np

class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass

def first(x):
    return x[0]

class DinoClassifier(KeyClassifier):
    def __init__(self, config=None):
        self.inputs = []
        self.outputs = []

        np.random.seed(42)  # For reproducibility
        self.input_size = 7
        self.hidden_size = 7
        self.output_size = 2
        
        if config is not None:
            self.W1, self.W2, self.b1, self.b2 = config
        else:
            self.initialize_weights_and_biases()
    
    def initialize_weights_and_biases(self):
        # Initialize weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases as zeros
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        sigmoid_pos = 1 / (1 + np.exp(-x[pos_mask]))
        sigmoid_neg = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        result = np.empty_like(x)
        result[pos_mask] = sigmoid_pos
        result[neg_mask] = sigmoid_neg
        return result
    
    def softmax(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        # Propagate inputs through the network
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs

    def backward(self, X, y, lr):
        # Compute gradients
        delta3 = self.probs
        delta3[range(np.array(X).shape[0]), y] -= 1
        delta2 = np.dot(delta3, np.array(self.W2).T) * (self.a1 * (1 - self.a1))

        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        dW1 = np.dot(np.array(X).T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update weights and biases
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def fit(self, X, y, epochs=100, lr=0.01):
        for epoch in range(epochs):
            # Forward propagation
            probs = self.forward(X)
            # Backpropagation
            self.backward(X, y, lr)

                
    def predict(self, X):
        # Predict the output
        probs = self.forward(X)
        predictions = np.argmax(probs, axis=1)
        return predictions
    
    def get_weights(self):
        # put w1, w2, b1, b2 in a list
        return [self.W1, self.W2, self.b1, self.b2]
    
        
    
    def enumerateObType(self, obType):
        # check if class of obType is __main__.SmallCactus
        name = obType.__class__.__name__
        if name == "SmallCactus":
            return 10
        elif name == "LargeCactus":
            return 200
        elif name == "Bird":
            return 3000
        else:
            return 10
    
    def keySelector(self,distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        
        x = np.array([distance/10, obHeight, speed, self.enumerateObType(obType), nextObDistance/10, nextObHeight, self.enumerateObType(nextObType)])
        x = x.reshape(1, -1)
        self.inputs.append(x[0])
        predictions = self.predict(x)
        action = ""
        if predictions[0] == 0:
            self.outputs.append(0)
            action = "K_DOWN"
        elif predictions[0] == 1:
            self.outputs.append(1)
            action = "K_UP"
        else:
            action = "K_NO"
        return action
    
    def save_weights(self, filename):
        # save weights and biases to a file
        np.save(filename, self.get_weights())