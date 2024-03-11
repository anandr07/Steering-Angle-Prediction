#%%
import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2 # Protocol buffer definition for the TensorFlow saver
import driving_data 
import model

LOGDIR = './save' # Directory path where the trained model and TensorBoard logs will be saved

sess = tf.InteractiveSession()

L2NormConst = 0.001 # L2 Regularization Parameter value

train_vars = tf.trainable_variables() # train_vars corresponds to the weights and biases of the neural network layers.

# Computes the mean squared error loss, which measures the average squared difference between the predicted output and the actual target output.
# Apply L2 Norm on all the weights in the NN (Multiplied by L2 Norm Constant)
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst 
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) # Using Adam Optimizer to minimize loss
sess.run(tf.initialize_all_variables()) # Initializes all variables in the TensorFlow graph before training.

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss) # creates a summary operation for logging the value of the loss function during training.
# Merge all summaries into a single op
merged_summary_op =  tf.summary.merge_all() # merges all the summaries created in the TensorFlow graph into a single operation.

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1) # Save the models

# op to write logs to Tensorboard
logs_path = './model_summary' 
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 30
batch_size = 100

# Train the model
# Train a neural network model for multiple epochs, updating parameters with mini-batch gradient descent.
# Logging loss and summary data for visualization in TensorBoard and saving checkpoints for model persistence.
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)): # Number of Images (45567) / Batch Size (100)
    xs, ys = driving_data.LoadTrainBatch(batch_size) 
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0: # Every 10 Batches
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # Write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

# %%
