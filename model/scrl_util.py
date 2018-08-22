self.log_p = self.Y * tf.log(self.a_pre)
self.log_lik = self.log_p * self.adv
self.loss = tf.reduce_mean(tf.reduce_sum(-self.log_lik, axis=1))
self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)