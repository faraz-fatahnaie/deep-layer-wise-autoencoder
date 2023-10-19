import tensorflow as tf
from functools import partial


class EQLv2(tf.keras.losses.Loss):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=2,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False, **kwargs):

        self.pred_class_logits = None
        self.gt_classes = None
        self.n_c = None
        self.n_i = None
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + tf.exp(-gamma * (x - mu)))

        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        super(EQLv2, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        target, cls_score = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)
        self.n_i, self.n_c = cls_score.shape[0], cls_score.shape[1]  # none 10
        self.gt_classes = target
        self.pred_class_logits = cls_score
        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)  # (None, 10)
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(target, cls_score)
        cls_loss = tf.reduce_sum(cls_loss * weight) / self.n_c

        # self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        self.collect_grad(cls_score, target, weight)
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = tf.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = tf.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = tf.abs(grad)
        pos_grad = tf.reduce_sum(grad * target * weight)
        neg_grad = tf.reduce_sum(grad * (1 - target) * weight)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = tf.zeros(self.num_classes)
            self._neg_grad = tf.zeros(self.num_classes)
            neg_w = tf.ones(self.n_c)
            pos_w = tf.ones(self.n_c)
        else:
            # the negative weight for objectiveness is always 1
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = tf.reshape(neg_w, shape=(1, -1))
            pos_w = tf.reshape(pos_w, shape=(1, -1))

        return pos_w, neg_w
