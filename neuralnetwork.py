import torch
import numpy as np




class NeuralNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense0 = torch.nn.Linear(in_features=10 , out_features=8 ,bias=True)
        self.activation = torch.nn.Tanh()
        self.last = torch.nn.Linear(in_features=8, out_features=1 , bias=False)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X= self.dense0(X)
        X = self.activation(X)
        return  self.last(X)
    
class Trainer(torch.nn.Module):
    
    
    def __init__(self, neuralnetwork : NeuralNetwork, optimizer: torch.optim.Optimizer, loss: torch.nn.Module , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nn = neuralnetwork
        self.optimizer = optimizer
        self.loss = loss
        assert self.optimizer.param_groups[0]['params'] == list(self.nn.parameters())



    def _permute(self, X : torch.Tensor, y : torch.Tensor ) -> torch.Tensor:
        assert X.shape[0] == y.shape[0], \
        f"To generate batches, X has {X.shape[0]} and y has {y.shape[0]} number of rows. Cannot generate batch"
        perm = np.random.permutation(X.shape[0])
        return X[perm] , y[perm]
    

    def _batch(self, X :  torch.Tensor, y :  torch.Tensor, batch_size : int ) -> tuple:
       
       
        assert X.shape[0] == y.shape[0], \
        f"To generate batches, X has {X.shape[0]} and y has {y.shape[0]} number of rows. Cannot generate batch"

        for steps in np.arange(0 , X.shape[0], batch_size):
            X_b, y_b = X[steps: steps + batch_size] , y[steps: steps + batch_size]

            yield X_b, y_b



    def train(self, X_train : torch.Tensor, y_train : torch.Tensor , X_test :  torch.Tensor , y_test :  torch.Tensor, epoch : int = 100 , batch_size: int = 20 ) -> None:
        
        for e in np.arange(epoch):
            print(f"-----------------------------------------------------EPOCH {e+1} ----------------------------------------------------")
            X_train , y_train = self._permute(X_train, y_train)

            batch = self._batch(X_train , y_train, batch_size)

            for ii, (X_b, y_b) in enumerate(batch):
                self.nn.zero_grad()
                out = self.nn.forward(X_b)
                loss = self.loss(out, y_b)
                loss.backward()
                self.optimizer.step()
                test_pred = self.nn.forward(X_test)
                loss = self.loss.forward(test_pred , y_test)
                print(f"[INFO] Epoch {e}, Batch {ii}: Batch Loss = {loss:.4f}")
        
        return

            
                
                    



            
    
    

if __name__ == '__main__':
    X  = np.loadtxt(fname="testX.csv", dtype=np.float32, skiprows=1, delimiter=',')
    y = np.loadtxt(fname="testY.csv", dtype=np.float32, skiprows=1, delimiter=',')

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    num_train = 18000
    X_train , X_test = X_t[:num_train, :].cuda(0) , X_t[num_train:].cuda(0)
    y_train , y_test = y_t[:num_train].view(-1,1).cuda(0) , y_t[num_train:].view(-1,1).cuda(0)

   
    nn = NeuralNetwork().cuda()
    optimizer = torch.optim.SGD(nn.parameters(),  lr=0.01, momentum=0.9)
    loss = torch.nn.MSELoss().cuda()
    trainer = Trainer(nn, optimizer, loss).cuda()
    trainer.train(X_train, y_train, X_test, y_test, batch_size=500)
