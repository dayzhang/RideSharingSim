# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense
# from simulator import settings
# from simulator.settings import FLAGS
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# # Standrad Implementation of DeepQNetworks "Parent Class"
# class DeepQNetwork(object):
#     # tf.compat.v1.disable_eager_execution()
#     def __init__(self, network_path=None):
#         self.sa_input, self.q_values, self.model = self.build_q_network()
#         # print(FLAGS.save_network_dir)
#         if not os.path.exists(FLAGS.save_network_dir):
#             os.makedirs(FLAGS.save_network_dir)
#         # To save the retrained model so it could be used to run the model as (pre-trained)
#         self.saver = tf.compat.v1.train.Saver(self.model.trainable_weights)
#         self.sess = tf.compat.v1.InteractiveSession()
#         self.sess.run(tf.compat.v1.global_variables_initializer())

#         if network_path:    # If previously constructed network is saved, load it
#             print("Net:", network_path)
#             self.load_network(network_path)

#     # Build the Q network with the required layers and number of features
#     def build_q_network(self):
#         sa_input = Input(shape=(settings.NUM_FEATURES,), dtype='float32')
#         x = Dense(100, activation='relu', name='dense_1')(sa_input)
#         x = Dense(100, activation='relu', name='dense_2')(x)
#         q_value = Dense(1, name='q_value')(x)
#         model = Model(inputs=sa_input, outputs=q_value)
#         return sa_input, q_value, model

#     # Restore the saved network to use for pretrain
#     def load_network(self, network_path):
#         # print("Net:", network_path)
#         self.saver.restore(self.sess, "/Users/mwadea/Documents/RideSharing/logs/tmp/networks/model-10000")
#         # self.saver.restore(self.sess, network_path)

#         print('Successfully loaded: ' + network_path)


#     def compute_q_values(self, s):
#         s_feature, a_features = s
#         q = self.q_values.eval(
#             feed_dict={
#                 self.sa_input: np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
#             })[:, 0]
#         return q

#     # Get action associated with max q-value
#     def get_action(self, q_values, amax):
#         if FLAGS.alpha > 0:
#             exp_q = np.exp((q_values - q_values[amax]) / FLAGS.alpha)
#             p = exp_q / exp_q.sum()
#             return np.random.choice(len(p), p=p)
#         else:
#             return amax

#         # Get price associated with max q-value
#     def get_price(self, q_values, amax):
#         if FLAGS.alpha > 0:
#             exp_q = np.exp((q_values - q_values[amax]) / FLAGS.alpha)
#             p = exp_q / exp_q.sum()
#             return np.random.choice(len(p), p=p)
#         else:
#             return amax

# # Learner Q-network used in trining mode
# class FittingDeepQNetwork(DeepQNetwork):

#     def __init__(self, network_path=None):
#         super().__init__(network_path)
#         model_weights = self.model.trainable_weights
#         # Create target network
#         self.target_sub_input, self.target_q_values, self.target_model = self.build_q_network()
#         target_model_weights = self.target_model.trainable_weights

#         # Define target network update operation
#         self.update_target_network = [target_model_weights[i].assign(model_weights[i]) for i in
#                                       range(len(target_model_weights))]

#         # Define loss and gradient update operation
#         self.y, self.loss, self.grad_update = self.build_training_op(model_weights)
#         self.sess.run(tf.compat.v1.global_variables_initializer())

#         # if load_network:
#         #     self.load_network()

#         # Initialize target network
#         self.sess.run(self.update_target_network)

#         self.n_steps = 0
#         self.epsilon = settings.INITIAL_EPSILON
#         self.epsilon_step = (settings.FINAL_EPSILON - settings.INITIAL_EPSILON) / settings.EXPLORATION_STEPS


#         for var in model_weights:
#             tf.compat.v1.summary.histogram(var.name, var)
#         self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
#         self.summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.save_summary_dir, self.sess.graph)

#     # Greedy Approach to get action with max q-value
#     def get_action(self, q_values, amax):
#         # e-greedy exploration
#         if self.epsilon > np.random.random():
#             return np.random.randint(len(q_values))
#         else:
#             return super().get_action(q_values, amax)

#     def get_fingerprint(self):
#         return self.n_steps, self.epsilon

#     # Calc target Q value based on State features and action features of next t
#     def compute_target_q_values(self, s):
#         s_feature, a_features = s
#         q = self.target_q_values.eval(
#             feed_dict={
#                 self.target_sub_input: np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
#             })[:, 0]
#         return q

#     def compute_target_value(self, s):
#         Q = self.compute_target_q_values(s)
#         amax = np.argmax(self.compute_q_values(s))
#         V = Q[amax]
#         if FLAGS.alpha > 0:
#             V += FLAGS.alpha * np.log(np.exp((Q - Q.max()) / FLAGS.alpha).sum())
#         return V

#     # Fitting the model using state action list and associated next state
#     def fit(self, sa_batch, y_batch):
#         loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
#             self.sa_input: np.array(sa_batch, dtype=np.float32),
#             self.y: np.array(y_batch, dtype=np.float32)
#         })
#         return loss

#     def run_cyclic_updates(self):
#         self.n_steps += 1
#         # Update target network
#         if self.n_steps % settings.TARGET_UPDATE_INTERVAL == 0:
#             self.sess.run(self.update_target_network)
#             print("Update target network")

#         # Save network
#         if self.n_steps % settings.SAVE_INTERVAL == 0:
#             save_path = self.saver.save(self.sess, os.path.join(FLAGS.save_network_dir, "model"), global_step=(self.n_steps))
#             print('Successfully saved: ' + save_path)

#         # Anneal epsilon linearly over time
#         if self.n_steps < settings.EXPLORATION_STEPS:
#             self.epsilon += self.epsilon_step

#     # Building the training optimizer (updating the gradient)
#     def build_training_op(self, q_network_weights):
#         # y = tf.compat.v1.placeholder(tf.float32, shape=(None))
#         # # q_value = tf.compat.v1.reduce_sum(self.q_values, reduction_indices=1)
#         # q_value = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(self.q_values)
#         # # loss = tf.compat.v1.losses.huber_loss(y, q_value)
#         # loss = tf.keras.losses.Huber()(y, q_value)
#         self.y = tf.keras.Input(shape=(1,), name="target_q")
#         q_value = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
#                                  name="q_value_sum")(self.q_values)

#         # Ensure `y` is shaped (batch, 1) as well (y might be a placeholder/KerasTensor)
#         y = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 1)), name="y_reshape")(self.y)

#         # Create a per-sample Huber loss using a Keras-compatible call (no direct tf function)
#         huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
#         per_sample_loss = tf.keras.layers.Lambda(lambda args: huber(args[0], args[1]),
#                                                 name="per_sample_huber")([y, q_value])
#         # Now reduce to a scalar loss (mean over batch), keep it as a Keras layer op
#         loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x), name="loss_mean")(per_sample_loss)


#         optimizer = tf.compat.v1.train.RMSPropOptimizer(settings.LEARNING_RATE, momentum=settings.MOMENTUM, epsilon=settings.MIN_GRAD)
#         with tf.compat.v1.get_default_graph().as_default():
#             loss_tensor = tf.identity(loss, name="loss_tensor")
#         grad_update = optimizer.minimize(loss_tensor, var_list=q_network_weights)

#         return y, loss, grad_update

#     def setup_summary(self):
#         avg_max_q = tf.compat.v1.Variable(0.)
#         tf.compat.v1.summary.scalar('Average_Max_Q', avg_max_q)
#         avg_loss = tf.compat.v1.Variable(0.)
#         tf.compat.v1.summary.scalar('Average_Loss', avg_loss)
#         summary_vars = [avg_max_q, avg_loss]
#         summary_placeholders = [tf.compat.v1.placeholder(tf.float32) for _ in range(len(summary_vars))]
#         update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
#         summary_op = tf.compat.v1.summary.merge_all()
#         return summary_placeholders, update_ops, summary_op

#     def write_summary(self, avg_loss, avg_q_max):
#         # tf.compat.v1.enable_eager_execution()
#         # print(self.update_ops)
#         # print(self.summary_placeholders)
#         stats = [avg_q_max, avg_loss]
#         for i in range(len(stats)):
#             self.sess.run(self.update_ops[i], feed_dict={
#                 self.summary_placeholders[i]: float(stats[i])
#             })
#         # print(self.summary_op)
#         summary_str = self.sess.run(self.summary_op)
#         # Write optimized avg loss, and avg q_max
#         self.summary_writer.add_summary(summary_str, self.n_steps)

# V2 Code

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from simulator import settings
from simulator.settings import FLAGS

# Standard Implementation of DeepQNetworks "Parent Class"
class DeepQNetwork(object):
    
    def __init__(self, network_path=None):
        self.model = self.build_q_network()
        
        if not os.path.exists(FLAGS.save_network_dir):
            os.makedirs(FLAGS.save_network_dir)
        
        # Create checkpoint for saving/loading
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, 
            FLAGS.save_network_dir, 
            max_to_keep=5
        )
        
        if network_path:  # If previously constructed network is saved, load it
            print("Net:", network_path)
            self.load_network(network_path)

    # Build the Q network with the required layers and number of features
    def build_q_network(self):
        sa_input = Input(shape=(settings.NUM_FEATURES,), dtype='float32', name='sa_input')
        x = Dense(100, activation='relu', name='dense_1')(sa_input)
        x = Dense(100, activation='relu', name='dense_2')(x)
        q_value = Dense(1, name='q_value')(x)
        model = Model(inputs=sa_input, outputs=q_value)
        return model

    # Restore the saved network to use for pretrain
    def load_network(self, network_path):
        # Try loading from checkpoint
        status = self.checkpoint.restore(network_path)
        print('Successfully loaded: ' + network_path)

    def compute_q_values(self, s):
        s_feature, a_features = s
        sa_batch = np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
        q = self.model(sa_batch, training=False).numpy()[:, 0]
        return q

    # Get action associated with max q-value
    def get_action(self, q_values, amax):
        if FLAGS.alpha > 0:
            exp_q = np.exp((q_values - q_values[amax]) / FLAGS.alpha)
            p = exp_q / exp_q.sum()
            return np.random.choice(len(p), p=p)
        else:
            return amax

    # Get price associated with max q-value
    def get_price(self, q_values, amax):
        if FLAGS.alpha > 0:
            exp_q = np.exp((q_values - q_values[amax]) / FLAGS.alpha)
            p = exp_q / exp_q.sum()
            return np.random.choice(len(p), p=p)
        else:
            return amax


# Learner Q-network used in training mode
class FittingDeepQNetwork(DeepQNetwork):

    def __init__(self, network_path=None):
        super().__init__(network_path)
        
        # Create target network
        self.target_model = self.build_q_network()
        
        # Initialize target network with same weights
        self.update_target_network()
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=settings.LEARNING_RATE,
            momentum=settings.MOMENTUM,
            epsilon=settings.MIN_GRAD
        )
        
        # Huber loss function
        self.huber_loss = tf.keras.losses.Huber()
        
        self.n_steps = 0
        self.epsilon = settings.INITIAL_EPSILON
        self.epsilon_step = (settings.FINAL_EPSILON - settings.INITIAL_EPSILON) / settings.EXPLORATION_STEPS
        
        # Setup TensorBoard logging
        self.summary_writer = tf.summary.create_file_writer(FLAGS.save_summary_dir)
        
        # Metrics for tracking
        self.avg_loss_metric = tf.keras.metrics.Mean(name='avg_loss')
        self.avg_q_metric = tf.keras.metrics.Mean(name='avg_q')

    def update_target_network(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())

    # Greedy Approach to get action with max q-value
    def get_action(self, q_values, amax):
        # e-greedy exploration
        if self.epsilon > np.random.random():
            return np.random.randint(len(q_values))
        else:
            return super().get_action(q_values, amax)

    def get_fingerprint(self):
        return self.n_steps, self.epsilon

    # Calc target Q value based on State features and action features of next t
    def compute_target_q_values(self, s):
        s_feature, a_features = s
        sa_batch = np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
        q = self.target_model(sa_batch, training=False).numpy()[:, 0]
        return q

    def compute_target_value(self, s):
        Q = self.compute_target_q_values(s)
        amax = np.argmax(self.compute_q_values(s))
        V = Q[amax]
        if FLAGS.alpha > 0:
            V += FLAGS.alpha * np.log(np.exp((Q - Q.max()) / FLAGS.alpha).sum())
        return V

    @tf.function
    def train_step(self, sa_batch, y_batch):
        """Single training step using GradientTape"""
        with tf.GradientTape() as tape:
            # Forward pass
            q_values = self.model(sa_batch, training=True)
            
            # Compute loss
            loss = self.huber_loss(y_batch, q_values)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    # Fitting the model using state action list and associated next state
    def fit(self, sa_batch, y_batch):
        sa_tensor = tf.convert_to_tensor(sa_batch, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        
        # Reshape y_tensor to match model output shape if needed
        if len(y_tensor.shape) == 1:
            y_tensor = tf.reshape(y_tensor, (-1, 1))
        
        loss = self.train_step(sa_tensor, y_tensor)
        
        return loss.numpy()

    def run_cyclic_updates(self):
        self.n_steps += 1
        
        # Update target network
        if self.n_steps % settings.TARGET_UPDATE_INTERVAL == 0:
            self.update_target_network()
            print("Update target network")

        # Save network
        if self.n_steps % settings.SAVE_INTERVAL == 0:
            save_path = self.checkpoint_manager.save(checkpoint_number=self.n_steps)
            print('Successfully saved: ' + save_path)

        # Anneal epsilon linearly over time
        if self.n_steps < settings.EXPLORATION_STEPS:
            self.epsilon += self.epsilon_step

    def write_summary(self, avg_loss, avg_q_max):
        """Write summary statistics to TensorBoard"""
        with self.summary_writer.as_default():
            tf.summary.scalar('Average_Loss', avg_loss, step=self.n_steps)
            tf.summary.scalar('Average_Max_Q', avg_q_max, step=self.n_steps)
            
            # Optionally log model weights histograms
            for var in self.model.trainable_variables:
                tf.summary.histogram(var.name, var, step=self.n_steps)
        
        self.summary_writer.flush()

    def save_model(self, path=None):
        """Save the model weights"""
        if path is None:
            path = os.path.join(FLAGS.save_network_dir, f"model_{self.n_steps}")
        self.model.save_weights(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model weights"""
        self.model.load_weights(path)
        print(f"Model loaded from {path}")