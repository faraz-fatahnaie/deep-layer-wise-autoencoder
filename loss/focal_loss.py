import tensorflow as tf


def focal_loss(alpha: float = 0.24, gamma: float = 2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -tf.reduce_sum(alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt))
        return loss

    return focal_loss_fixed
