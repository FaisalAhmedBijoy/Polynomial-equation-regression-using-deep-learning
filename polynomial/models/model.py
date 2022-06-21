import tensorflow as tf


def define_model_architecture():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=200,input_shape=[1]))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(units=100))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(units=1))
    model.summary()
    return model

if __name__=='__main__':
    model=define_model_architecture()
