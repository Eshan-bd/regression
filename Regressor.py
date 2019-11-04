import numpy as np

class Regressor:
    # Learning Rate
    alpha = 1
    ter_point = 0.0000001
    cost_arr = []
   
    def fit(self, pX, p_y):
        # add a column of zeroes for Xo
        self.X = np.hstack( (np.ones((pX.shape[0], 1)), pX) )
        self.y = np.copy(p_y)

        # coefficent of features
        self.theta = np.zeros((self.X.shape[1], 1))

        # m = number of samples
        self.m = self.X.shape[0]
        self.gradient_descent()
    
    def gradient_descent(self):
        theta = self.theta
        h = self.hyp()

        theta0, theta1 = theta[0], theta[1]

        # Gradient Descent for theta0
        diff0 = h - self.y
        cost0 = diff0.sum() / self.m
        theta0 -= self.alpha * cost0

        #Gradient Descent for theta1
        diff1 = (h - self.y) * self.X
        cost1 = diff1.sum() / self.m
        theta1 -= self.alpha * cost1
        
        theta[0], theta[1] = theta0, theta1
        h = self.hyp()
        # Cost function
        cost = np.square(h - self.y).sum() / self.m
        self.cost_arr.append(cost)
        self.update()


    def update(self):
        cost_arr = self.cost_arr
        i = len(cost_arr) - 1

        if i<=0 or cost_arr[i-1]-cost_arr[i] > self.ter_point:
            self.gradient_descent()

    def predict(self):
        return self.X.dot(self.theta)

    # Hypothesis
    def hyp(self):
        return self.X.dot(self.theta)