import tensorflow as tf

# TODO: refactor this
@tf.function
def product(tensors, n_tensors):
    # inspired by
    # https://stackoverflow.com/questions/47132665/cartesian-product-in-tensorflow/47133461
    t0 = tensors[0]
    t1 = tensors[1]
    tile_0 = tf.tile(tf.expand_dims(t0, 1), [1, tf.shape(t1)[0]])
    tile_0 = tf.expand_dims(tile_0, 2)
    tile_1 = tf.tile(tf.expand_dims(t1, 0), [tf.shape(t0)[0], 1])
    tile_1 = tf.expand_dims(tile_1, 2)

    cartesian_product = tf.reshape(tf.concat([tile_0, tile_1], axis=2), (-1, 2))
    if n_tensors == 3:
        t2 = tensors[2]

        tile_2 = tf.tile(tf.expand_dims(t2, 1), [1, tf.shape(cartesian_product)[0]])
        tile_2 = tf.expand_dims(tile_2, 2)
        tile_cartesian = tf.tile(tf.expand_dims(cartesian_product, 0), [tf.shape(t2)[0], 1, 1])
        cartesian_product = tf.reshape(tf.concat([tile_2, tile_cartesian], axis=2), (-1, 3))
    return cartesian_product
