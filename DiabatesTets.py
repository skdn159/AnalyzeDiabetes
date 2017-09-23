import tensorflow as tf
import numpy as np

#xy = np.loadtxt("3year.csv", delimiter=',',dtype=np.float)
xy = np.loadtxt("3year.csv", delimiter=',',dtype=np.float)

x_data = xy[:,1:-1]
y_data = xy[:,[-1]]

certi = np.loadtxt("certi.csv", delimiter=',',dtype=np.float)
cer_x = certi[:,1:-1] 
cer_y = certi[:,[-1]] 

#keepProb = 0.7
#batch_size = 100
learning_rate = 0.001
training_epochs = 3

x = tf.placeholder(tf.float32, shape = [None,24])
y = tf.placeholder(tf.int32, shape = [None,1])


###
'''
w1 = tf.Variable(tf.random_normal([24,24]))
b1 = tf.Variable(tf.random_normal([24]))

w2 = tf.Variable(tf.random_normal([24,1]))
b2 = tf.Variable(tf.random_normal([1]))

L1 = tf.add(tf.matmul(x,w1),b1)
L1 = tf.nn.relu(L1)
'''
###
'''
hypo = tf.sigmoid(tf.matmul(L1,w2) + b2)
#cost = -tf.reduce_mean(y * tf.log(hypo) + (1 - y) * tf.log(1 - hypo))
#cost = tf.reduce_sum(tf.square(hypo - y))
cost = tf.reduce_mean(tf.square(hypo - y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
'''

nb_classes = 2
y_one_hot = tf.one_hot(y,nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1,nb_classes])

w1 = tf.Variable(tf.random_normal([24,36]),name='weight1')
b1= tf.Variable(tf.random_normal([36]), name = 'bias1')
L1 = tf.nn.relu(tf.matmul(x,w1)+b1)

w2 = tf.Variable(tf.random_normal([36,36]),name='weight2')
b2= tf.Variable(tf.random_normal([36]), name = 'bias2')
L2 = tf.nn.relu(tf.matmul(L1,w2)+b2)

w3 = tf.Variable(tf.random_normal([36,nb_classes]),name='weight3')
b3= tf.Variable(tf.random_normal([nb_classes]), name = 'bias3')
L3 = tf.matmul(L2,w3)+b3

logit = L3
hypo = tf.nn.softmax(logit)
#hypo = tf.nn.softmax(tf.matmul(L1,w2)+b2)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits= logit , labels= y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

prediction = tf.arg_max(hypo,1)
correct_Prediction = tf.equal(prediction,tf.arg_max(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_Prediction,tf.float32))


'''
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo),axis =1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


is_correct =tf.equal(tf.arg_max(hypo,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(2000):
            c, _ = sess.run([cost, optimizer], feed_dict={x:x_data, y:y_data})
            
            if i % 100 == 0:
                print(i, "  ", c)

    #probb = sess.run(hypo, feed_dict = {x:cer_x} )
    #pred = sess.run(prediction, feed_dict = {x:cer_x})

    probb = sess.run(hypo, feed_dict = {x:cer_x} )
    pred = sess.run(prediction, feed_dict = {x:cer_x})


    idx= 0
    for p,y, pp in zip(pred, cer_y.flatten(),probb):
        print ("({})[{}] Prediction:{} True y: {}, pp:{}".format(idx,p==int(y),p,int(y),pp))
        idx +=1

    # Accuracy report
    '''
    h, c, a = sess.run([hypo, predicted, accuracy],
                       feed_dict={x: x_data, y: y_data})
    print("epoch =",epoch, "\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    '''
