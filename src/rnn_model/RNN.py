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
        if self.embed_size >0 and self.stock_count>1:
            self.embed_matrix = tf.Variable(tf.random_uniform([self.stock_count,self.embed_size],-1.0,1.0),name="embed_matrix")

            stacked_symbols = tf.tile(self.symbols,[1,self.num_steps], name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix,stacked_symbols)

            self.inputs_with_embeds = tf.concat([self.inputs,stacked_embeds], axis=2, name='inputs_wit_embed')
            self.embed_matrix_summ = tf.summary.histogram('embed_matrix', self.embed_matrix)
        else:
            self.inputs_with_embeds = tf.identity(self.inputs)
            self.embed_matrix_summ = None
        print("inputs.shape:", self.inputs.shape)
        print("inputs_with_embed.shape:", self.inputs_with_embed.shape)

        val, state_ = tf.nn.dynamic_rnn(cell,self.inputs_with_embeds,dtype=tf.float32, scope = 'dynamix_rnn')
        val = tf.transpose(val,[1,0,2])

        last = tf.gather(val, int(val.get_shape()[0])-1, name='lstm_state')
        ws = tf.Variable(tf.truncated_normal([self.lstm_size,self.input_size]), name='w')
        bias = tf.Variable(tf.constant(0.1,shape=[self.input_size]), name='b')
        self.pred = tf.matmul(last,ws)+bias

        self.last_sum = tf.summary.histogram('lstm_state', last)
        self.w_sum = tf.summary.histogram('w',ws)
        self.b_sum = tf.summary.histogram('b',bias)
        self.pred_summ = tf.summary.histogram('pred', self.pred)

        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name='loss_mse_train')
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name='rmsprop_optim')

        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name='loss_mse_test')

        self.loss_sum = tf.summary.scalar('loss_mse_train', self.loss)
        self.loss_test_sum = tf.summary.scalar('loss_mse_test', self.loss_test)
        self.learning_rate_sum = tf.summary.scalar('learning_rate', self.learning_rate)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()



