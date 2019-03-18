import tensorflow as tf

class RNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size = 128,
                 num_layers = 1,
                 num_steps = 30,
                 input_size = 1,
                 embed_size = None,
                 logs_dir = 'logs',
                 plots_dir = 'images'):
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size>0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        self.learning_rate = tf.placeholder(tf.float32, None, name = 'learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

        self.symbols = tf.placeholder(tf.float32,[None,1], name='stock_labels')

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None,self.input_size], name='targets')

        def create_one_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.nn.rnn_cell.MultiRNNCell([create_one_cell() for _ in range(self.num_layers)], state_is_tuple=True) if self.num_layers>1 else create_one_cell()


