
import tensorflow as tf

groups = tf.constant([[0,1,2,3],[4,5,6,7]])

arr = tf.constant([[10,0],[20,0],[30,0],[40,0],[50,0],[60,0],[70,0],[80,0]])

output = tf.gather(arr, groups)
print(output)