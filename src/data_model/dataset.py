from utils.DataImporter import import_data
import numpy as np


class DataSet():
    def __init__(self, filename, input_size = 1, num_steps = 30):
        self.input_size = input_size
        self.num_steps = num_steps
        self.filename = filename

        raw_data = import_data(filename)
        raw_sequence = raw_data['Close'].tolist()
        self.seq = np.array(raw_sequence)

    def prepare_data(self,seq):
        seq = [np.array(seq[i*self.input_size : (i+1)*self.input_size])
               for i in range(len(seq)//self.input_size)]

        norm = [seq[0]/seq[0][0]-1.0] +[current/seq[i][-1] -1.0 for current,i in enumerate(seq[1:])]
        X = np.array([norm[i:i+self.num_steps] for  i in range(len(norm)-self.num_steps)])
        y = np.array([norm[i+self.num_steps] for i in range])




