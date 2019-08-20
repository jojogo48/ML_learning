import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(self):
        pass


    def load_data(self, x, y):
        self.x = x
        self.y = y


    def sigmoid(self, z):
        return 1 / ( 1 + exp(-z))


    def accuracy(self, Y_pred, y_label):
        return np.sum(Y_pred == y_label) / len(Y_pred)


    def cross_entropy(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-8, 1.0 - 1e-8)
        return - np.sum( y_label * log(y_pred) + (1 - y_label) * log(1 - y_pred)) / len(y_pred) 

    
    def get_prob(self, w, x, b):
        return self.sigmoid(x.dot(w) + b)


    def infer(self, y_pred):
        return np.round(y_pred)
    

    def gradient(self, y_pred, y_label, x):
        w_grad = -(x.T).dot(y_label - y_pred)
        b_grad = -np.sum(y_label - y_pred)
        return w_grad, b_grad


    def z_score_normal(self):
        x_mean = np.mean(self.x, axis=0)
        x_std = np.std(self.x, axis=0)
        self.x = (self.x - x_mean) / x_std


    def max_min_normal(self):
        x_max = np.max(self.x, axis=0)
        x_min = np.min(self.x, axis=0)
        self.x = (self.x - x_min) / (x_max - x_min)

    def shuffle(self):
        randomlization = np.arange(self.x.shape[0])
        np.random.shuffle(randomlization)
        self.x = self.x[randomlization]
        self.y = self.y[randomlization]


    def train(self, optimizer='Adam', lr=0.02, batch_size=32, epoch=32, decay=0):
        self.w = np.zeros(shape=(self.x.shape[1], 1))
        self.b = 0
        
        if optimizer == 'Adam':
            m_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            m_b_grad = 0
            
            v_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            v_b_grad = 0


            b1 = 0.9
            b2 = 0.999
            e = 1e-8

            t = 0
        elif optimizer == 'Adagrad':
            total_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            total_b_grad = 0
            e = 1e-8

        loss_his = []

        for i in range(epoch):

            self.shuffle()
            batch_num = int(len(self.x)/batch_size)
            
            

            for j in range(batch_num):
                
                x_train = self.x[j*batch_size:(j+1)*batch_size]
                y_train = self.y[j*batch_size:(j+1)*batch_size].reshape(batch_size, 1)
                    
                y_pred = self.get_prob(self.w, x_train, self.b)
                Y_pred = self.infer(y_pred)
                

                w_grad, b_grad = self.gradient(y_pred, y_train, x_train)

                if optimizer == 'Adam':
                    t += 1     

                    m_w_grad = b1 * m_w_grad + (1 - b1) * w_grad
                    m_b_grad = b1 * m_b_grad + (1 - b1) * b_grad

                    v_w_grad = b2 * v_w_grad + (1 - b2) * w_grad**2
                    v_b_grad = b2 * v_b_grad + (1 - b2) * b_grad**2

                    m_w_unbias = m_w_grad / (1 - b1**t)
                    m_b_unbias = m_b_grad / (1 - b1**t)

                    v_w_unbias = v_w_grad / (1 - b2**t)
                    v_b_unbias = v_b_grad / (1 - b2**t)
                    

                    self.w =((1 - decay) * self.w) - lr * m_w_unbias / (np.sqrt(v_w_unbias) + e)
                    self.b =((1 - decay) * self.b) - lr * m_b_unbias / (np.sqrt(v_b_unbias) + e)
                elif optimizer == 'Adagrad':
                    total_w_grad += w_grad**2
                    total_b_grad += b_grad**2

                    self.w -= lr * w_grad / (np.sqrt(total_w_grad) + e)
                    self.b -= lr * b_grad / (np.sqrt(total_b_grad) + e)
                    

                ac = self.accuracy(Y_pred, y_train)
                loss = self.cross_entropy(y_pred, y_train)
                print(f'loss : {loss:<10f} ac : {ac:<10f}')

                loss_his.append(loss)
                
        plt.plot(loss_his)
        plt.show()
                
        return self.w ,self.b        


