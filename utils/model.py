class Perceptron:
    def __init__(self,eta,epochs):
        self.weights=np.random.randn(3)*1e-4
        print(f"Initial weights before training: {self.weights}")
        self.eta=eta
        self.epochs=epochs

    def activateFunction(self,inputs,weights):
        z=np.dot(input,weights)
        return np.where(z>0,1,0)

    def fit(self,x,y):
        self.x=x
        self.y=y
        x_with_bias=np.c_[self.x,-np.ones((len(self.x),1))]
        print(f"X with bias:{x_with_bias}")
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch:{epoch}")
            print("--"*10)
            y_hat=self.activationFunction(X_with_bias,self.weights)
            print(f"predicted value after forward propagation:{y_hat}")
            self.weights=self.weights+self.eta*np.dot(x_with_bias.T,self.error)
            print(f"error:{self.error}")
            print("updated weights after epoch:{epoch}/{self.epochs}: {self.weights")
            print("######"*10)
    
    def predict(self,x):
        x_with_bias=np.c_[self.x,-np.ones((len(x),1))]
        return self.activateFunction(x_with_bias,self.weights)
    
    def total_loss(self):
        total_loss=np.sum(self.error)
        print(f"total loss:{total_loss}")
        return total_loss

    def prepare_data(df):
        x=df.drop("y",axis=1)
        y=df["y"]
        return x,y

