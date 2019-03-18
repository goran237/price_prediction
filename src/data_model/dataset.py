from utils.DataImporter import import_data
import numpy as np


class DataSet():
    def __init__(self, filename, input_size = 1, num_steps = 30, test_ratio = 0.1):
        self.input_size = input_size
        self.num_steps = num_steps
        self.filename = filename
        self.test_ratio = test_ratio

        raw_data = import_data(filename)
        raw_sequence = raw_data['Close'].tolist()
        self.seq = np.array(raw_sequence)
        self.train_X,self.train_y,self.test_X,self.test_y = self.prepare_data(seq)

    def prepare_data(self,seq):
        seq = [np.array(seq[i*self.input_size : (i+1)*self.input_size])
               for i in range(len(seq)//self.input_size)]

        norm = [seq[0]/seq[0][0]-1.0] +[current/seq[i][-1] -1.0 for current,i in enumerate(seq[1:])]
        X = np.array([norm[i:i+self.num_steps] for  i in range(len(norm)-self.num_steps)])
        y = np.array([norm[i+self.num_steps] for i in range(len(norm)-self.num_steps)])

        train_size = int(len(X)*(1.0-self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return  train_X,train_y,test_X,test_y

    def generate_one_epoch(self,batch_size):
        num_batches = int(len(self.train_X)//batch_size)
        if batch_size * num_batches < len(self.train_X):
            num_batches +=1
        batch_indices = range(num_batches)
        for j in batch_indices:
            batch_X = self.train_X[j*batch_size:(j+1)*batch_size]
            batch_y = self.train_y[j*batch_size:(j+1)*batch_size]
            assert set(map(len,batch_X)) =={self.num_steps}
            yield  batch_X, batch_y




