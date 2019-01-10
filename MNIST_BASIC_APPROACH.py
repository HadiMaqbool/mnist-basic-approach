#Importing the TensorFlow Library
import tensorflow as tf

#Importing the MNIST-Data Set from tensorflow built in function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

# The number of exaples in train, test and validation dataset
print(mnist.train.num_examples)
print(mnist.test.num_examples)
print(mnist.validation.num_examples)

#Data-Sample visulaization
import matplotlib.pyplot as plt
sample_img = mnist.train.images[1].reshape(28,28)
#plt.imshow(sample_img, cmap ='gist_gray')

#Create Placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

#Create Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.ones([10]))

#Create Graph Operations
y = tf.matmul(x,W) + b

#Create Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y))

#Create Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

#Create the global Variable initializer
init = tf.global_variables_initializer()

#Create a Session
with tf.Session() as sess:
     
      sess.run(init)
      
      for step in range(1000):
          
          batch_x, batch_y = mnist.train.next_batch(100)
          # Here x and y_true inputs to the feed dictionary are the placeholder created
          sess.run(train, feed_dict = {x: batch_x, y_true: batch_y })
         
        # Evaluating the model 
      correct_prediction = tf.equal(tf.argmax(y, 1) , tf.argmax(y_true, 1))  
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     
      print("results")
      print(sess.run(accuracy, feed_dict ={x:mnist.test.images, y_true : mnist.test.labels} ))  
#complete
