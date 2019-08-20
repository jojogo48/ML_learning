import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        pass


    def load_data(self, x, y):
        self.x = x
        self.y = y


    def z_score_normal(self):
        x_mean = np.mean(self.x, axis=0)
        x_std = np.std(self.x, axis=0)
        self.x = (self.x - x_mean) / x_std


    def max_min_normal(self):
        x_max = np.max(self.x, axis=0)
        x_min = np.min(self.x, axis=0)
        self.x = (self.x - x_min) / (x_max - X_min)


    def shuffle(self):
        randomlization = np.arange(len(self.x))
        np.random.shuffle(randomlization)
        self.x = self.x[randomlization]
        self.y = self.y[randomlization]

    
    def loss_mse(self, y_pred, y_label):
        return np.sum((y_pred - y_label)**2) / len(y_pred)


    def gradient(self, y_pred, y_label, x_train):
        w_grad = (-2) * (x_train.T).dot(y_label - y_pred)
        b_grad = (-2) * np.sum((y_label - y_pred))
        return w_grad, b_grad
        

    def train(self, optimizer='Adam', loss_fun='mse', lr=0.01, batch_size=32, epoch=2000):
        
        self.w = np.zeros(shape=(self.x.shape[1],1))
        self.b = np.zeros(1,)
        
        loss_history = []
        if optimizer == 'Adam':
            self.b1 = 0.9
            self.b2 = 0.999
            self.e = 1e-8
            self.t = 0

            self.m_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            self.m_b_grad = 0
            self.v_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            self.v_b_grad = 0
        if optimizer == 'Adagrad':
            total_w_grad = np.zeros(shape=(self.x.shape[1], 1))
            total_b_grad = 0
        for i in range(epoch):
            
            self.shuffle()
            batch_num = int(len(self.x) / batch_size)
            
            #print('epoch: ',i,' --'*10)

            for j in range(batch_num):
                x_train = self.x[j*batch_size: (j+1)*batch_size]
                y_train = self.y[j*batch_size: (j+1)*batch_size]
                
                y_pred = (x_train.dot(self.w) + self.b)

                if optimizer == 'Adam':
                    w_grad, b_grad = self.gradient(y_pred, y_train, x_train)
                    self.t += 1

                    self.m_w_grad = self.b1 * self.m_w_grad + (1 - self.b1) * w_grad
                    self.m_b_grad = self.b1 * self.m_b_grad + (1 - self.b1) * b_grad
                    
                    self.v_w_grad = self.b2 * self.v_w_grad + (1 - self.b2) * w_grad**2
                    self.v_b_grad = self.b2 * self.v_b_grad + (1 - self.b2) * b_grad**2

                    
                    m_w_unbias = self.m_w_grad / (1 - self.b1**self.t)
                    m_b_unbias = self.m_b_grad / (1 - self.b1**self.t)

                    v_w_unbias = self.v_w_grad / (1 - self.b2**self.t)
                    v_b_unbias = self.v_b_grad / (1 - self.b2**self.t)
 
                    self.w -= lr * (m_w_unbias/((np.sqrt(v_w_unbias))+self.e))
                    self.b -= lr * (m_b_unbias/(np.sqrt(v_b_unbias)+self.e))
                elif optimizer == 'Adagrad':
                    w_grad, b_grad = self.gradient(y_pred, y_train, x_train)
                    total_w_grad += w_grad**2
                    total_b_grad += b_grad**2

                    self.w -=lr * w_grad / (np.sqrt(total_w_grad))
                    self.b -=lr * b_grad / (np.sqrt(total_b_grad))
                if loss_fun == 'mse':
                    loss = self.loss_mse(y_pred, y_train)
                

                print('loss :', loss)
                loss_history.append(loss)

        plt.plot(loss_history)
        plt.show()
        return self.w, self.b
    

    def closed_form_solution(self):
        x_numb = self.x.shape[0]
        x = np.concatenate((self.x,np.ones(shape=(x_numb,1))) ,axis=1)
        solution = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(self.y)
        return solution[:-1], solution[-1]

