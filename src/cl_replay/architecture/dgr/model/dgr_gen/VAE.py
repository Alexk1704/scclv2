import copy
import os
import time
import math
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
# from tqdm import tqdm

from cl_replay.api.utils import log


class VAE(tf.keras.Model):
    
    
    def __init__(self, **kwargs):
        super().__init__()
        self.dgr_model = kwargs.get('dgr_model')    
        self.encoder = None
        self.decoder = None
        
        self.data_dim       = kwargs.get('input_size')
        self.label_dim      = kwargs.get('num_classes')
        self.batch_size     = kwargs.get('batch_size')
        self.vae_epochs     = kwargs.get('vae_epochs')
        
        self.recon_loss     = kwargs.get('recon_loss')
        self.latent_dim     = kwargs.get('latent_dim')
        
        self.enc_cond_input = kwargs.get('enc_cond_input')
        self.dec_cond_input = kwargs.get('dec_cond_input')
        
        self.vae_beta       = kwargs.get('vae_beta')
        
        self.vae_epsilon    = kwargs.get('vae_epsilon')
        self.adam_beta1     = kwargs.get('adam_beta1')
        self.adam_beta2     = kwargs.get('adam_beta2')
        
        self.optimizer      = tf.keras.optimizers.Adam(self.vae_epsilon, self.adam_beta1, self.adam_beta2)
        
        self.vis_path       = kwargs.get('vis_path')

        self.amnesiac       = kwargs.get('amnesiac', 'no')
        if self.amnesiac == 'yes':
            self.sa_lambda  = kwargs.get('sa_lambda', 100.)
            self.sa_gamma   = kwargs.get('sa_gamma', 1.)
            self.num_tasks  = kwargs.get('num_tasks', 1)
            
            self.init_tf_variables()


    def init_tf_variables(self):
        # self.ewc_storage    = {i: [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables] for i in range(1, self.num_tasks)}
        # self.fims           = {i: [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables] for i in range(1, self.num_tasks)}
        # self.coeffs         = {i: tf.Variable(0.0, trainable=False) for i in range(1, self.num_tasks)}
        self.ewc_storage    = [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables]
        self.fim            = [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables]
        
        self.fim_acc        = [tf.Variable(initial_value=var*0., trainable=False, name=f"acc{i}") for i, var in enumerate(self.trainable_variables)]
        self.lambda_        = tf.Variable(self.sa_lambda, trainable=False)
        self.gamma_         = tf.Variable(self.sa_gamma, trainable=False)


    def compute_loss(self, recon_x, real_x, mean, logvar, z, sw=None):
        # -----
        # NOTE Bernoulli dist. BCE+KLD:
        # real_x  = tf.reshape(real_x, shape=(-1, tf.reduce_prod(real_x.shape[1:])))
        # recon_x = tf.reshape(recon_x, shape=(-1, tf.reduce_prod(recon_x.shape[1:])))

        # bce     = tf.reduce_mean(tf.reduce_sum(tf.keras.ops.binary_crossentropy(target=real_x, output=recon_x), axis=-1)) 
        # kld     = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar)))

        # return bce + kld
        # -----
        # NOTE: Gaussian dist., Sigmoid Cross-Entropy
        cross_ent_loss  = tf.nn.sigmoid_cross_entropy_with_logits(logits=recon_x, labels=real_x)
            
        logpx_z     = -tf.reduce_sum(cross_ent_loss, axis=[1, 2, 3])
        
        if type(sw) != type(None):  # opt. sample weights
            if sw.any(): logpx_z *= sw  
        
        # Single sample Monte-Carlo estimate of the expectation
        logpz       = self.log_normal_pdf(z, 0., 0.)
        logqz_x     = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)  # NLL


    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    

    def encode(self, xs, ys):
        if self.enc_cond_input == 'yes':
            mean, logvar = self.encoder([xs, ys])
        else:
            mean, logvar = self.encoder(xs)
        
        return mean, logvar


    def reparameterize(self, mean, logvar):
        """ 
        Reparameterization trick.
        - Sample randomly from Gaussian dist., parameterized by eps (mean) & Sigma (std.). 
        """
        std = tf.exp(logvar * .5)
        eps = tf.random.normal(shape=mean.shape, stddev=std)
        return eps * std + mean


    def decode(self, zs, ys, apply_sigmoid=False):
        if self.dec_cond_input == 'yes':
            logits = self.decoder([zs, ys])
        else:
            logits = self.decoder(zs)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        else:
            return logits


    def fit(self, *args, **kwargs):
        """ assume that first argument is an iterator object """
        steps_per_epoch = kwargs.get("steps_per_epoch", 1)
        epochs          = kwargs.get("epochs", 1)
        max_steps       = epochs * steps_per_epoch
        
        log_metrics     = kwargs.get("callbacks")[0]
        log_metrics.set_model(self.dgr_model)
        log_metrics.on_train_begin()
        
        log.debug(f'training vae-gen for {epochs} epochs with {steps_per_epoch} steps...')
        epoch = 0

        for i, (x, y, sample_weights) in enumerate(args[0], start=1):
            self.train_step(x, y, sample_weights)
            log_metrics.on_batch_end(batch=i)

            if i % steps_per_epoch == 0:
                log.info(
                    f'epoch {epoch} step {i}\t' + 
                    f'vae_loss\t{self.dgr_model.metrics[0].result()}\t' +
                    f'step_time\t{self.dgr_model.metrics[1].result()}')
                epoch += 1 
                log_metrics.on_epoch_end(epoch)
            
            if i == max_steps: break

        log_metrics.custom_name = 'encoder'
        log_metrics.on_train_end()
        log_metrics.current_task -= 1


    def train_step(self, xs, ys, sample_weights, **kwargs):
        xs = tf.convert_to_tensor(xs)
        ys = tf.convert_to_tensor(ys)
        
        t1 = tf.timestamp(name='t1')

        with tf.GradientTape() as tape:
            mean, logvar    = self.encode(xs, ys)
            z               = self.reparameterize(mean, logvar)
            recon_x         = self.decode(z, ys, apply_sigmoid=False)  # NOTE: apply_sigmoid?
            loss            = self.compute_loss(recon_x, xs, mean, logvar, z, sample_weights)

        gradients = tape.gradient(loss, self.trainable_variables) # compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # update weights
    
        t2 = tf.timestamp(name='t2')
        delta = tf.subtract(t2, t1)
        
        self.dgr_model.metrics[0].update_state(loss)
        self.dgr_model.metrics[1].update_state(delta)


    def sample(self, eps=None, batch_size=100, scalar_classes=None):
        if eps is None:
            eps     = tf.random.normal(shape=(batch_size, self.latent_dim))
        if scalar_classes is not None:
            rnd_ys  = np.random.choice(scalar_classes, size=batch_size)
            tmp     = tf.eye(self.label_dim, self.label_dim)
            ys      = tf.gather(tmp, rnd_ys)
            return self.decode(eps, ys, apply_sigmoid=True), ys
        else:
            return self.decode(eps, apply_sigmoid=True)


    def set_parameters(self, **kwargs):
        self.current_task = kwargs.get('current_task', -1)
        self.prev_tasks = kwargs.get('prev_tasks', [])


    def save_weights(self, *args, **kwargs): # FIXME: serialization
        pass


    def load_weights(self, *args, **kwargs): # FIXME: serialization
        pass


    def visualize_samples(self, num_samples, classes, filename):
        if classes == []:
            classes = np.arange(self.label_dim)
        sampled, labels = self.sample(eps=None, batch_size=num_samples, scalar_classes=classes)

        if self.vis_path is None: return
        else:
            save_path = f'{self.vis_path}/gen/'
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_path = f'{save_path}{filename}.png'
        
        # ax_size = math.ceil(math.sqrt(sampled.shape[0]))
        ax_size = int(math.sqrt(sampled.shape[0]))
        max_samples = ax_size**2
        fig, axes = plt.subplots(ax_size, ax_size, figsize=(ax_size, ax_size))
        
        for i in range(max_samples):
            img_data = sampled[i]
            label = labels[i]
            ax = axes[i//ax_size, i%ax_size]
            if tf.is_tensor(img_data): img_data = img_data.numpy()
            ax.imshow(img_data.squeeze(), interpolation='nearest', cmap='viridis')
            ax.set_axis_off(); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_aspect('equal')
            if label is not None:
                if tf.is_tensor(label): label = label.numpy()
                ax.set_title(f'label: {label.argmax()}')
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.axis('off')
        plt.savefig(save_path, transparent=False, bbox_inches='tight')
        plt.close('all')


    def copy_weights(self, source, target):
        for source_layer, target_layer in zip(source, target):
            source_weights = source_layer.get_weights()
            target_layer.set_weights(source_weights)
            target_weights = target_layer.get_weights()

            if source_weights and all(tf.nest.map_structure(np.array_equal, source_weights, target_weights)):
                log.debug(f'copying weights from: {self.decoder}-{source_layer.name} -> {self.decoder_copy}-{target_layer.name}')


    # ---------------> SA
    # NOTE: assumes a separate forgetting phase!
    def forget_training(self, num_iters, batch_size, forget_classes, preserved_classes):
        # copy weights of current decoder to the clone for SA
        self.copy_weights(source=self.decoder.layers, target=self.decoder_copy.layers)

        if not batch_size: batch_size = self.batch_size
        
        log.debug(f'performing SA forget training for i = {num_iters} & mbs = {batch_size}\n' +
                  f'\t\tclasses to forget: {forget_classes}\n' +
                  f'\t\tclasses to preserve: {preserved_classes}')

        for i in range(math.ceil(num_iters/batch_size)):
            t1 = time.time()

            # -- generate class labels for classes to keep
            ys_pres_    = np.random.choice(preserved_classes, size=batch_size)
            tmp         = tf.eye(batch_size, self.label_dim)
            ys_pres     = tf.gather(tmp, ys_pres_)
            z_pres      = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            # -- generate class labels for classes to drop
            ys_forget_  = np.random.choice(forget_classes, size=batch_size)
            tmp         = tf.eye(batch_size, self.label_dim)
            ys_forget   = tf.gather(tmp, ys_forget_)
            # generate a random image [0,1] (noise) to corrupt the VAE loss
            gen_forget  = np.random.normal(size=(batch_size, self.data_dim[0], self.data_dim[1], self.data_dim[2]))  
            gen_forget  = np.divide(  # normalize pixel values
                gen_forget - gen_forget.min(),
                gen_forget.max() - gen_forget.min()
            )
            gen_forget  = tf.convert_to_tensor(gen_forget, dtype=tf.float32)

            # use decoder copy to transform noise into real samples for the classes we want to preserve
            gen_pres    = tf.sigmoid(self.decoder_copy([z_pres, ys_pres]))  # NOTE: apply_sigmoid? yes, since we are actually sampling here.
        
            with tf.GradientTape() as tape:
                # -- corrupt loss
                mean, logvar    = self.encode(gen_forget, ys_forget)
                z               = self.reparameterize(mean, logvar)
                recon_x         = self.decode(z, ys_forget, apply_sigmoid=False)  # NOTE: apply_sigmoid? depends!
                loss            = self.compute_loss(recon_x, gen_forget, mean, logvar, z, None)
                # print("corrupted loss: ", loss)
                
                # -- contrastive loss
                mean_, logvar_   = self.encode(gen_pres, ys_pres)
                z_               = self.reparameterize(mean_, logvar_)
                recon_x_         = self.decode(z_, ys_pres, apply_sigmoid=False)  # NOTE: apply_sigmoid? depends!
                loss            += self.gamma_ * self.compute_loss(recon_x_, gen_pres, mean_, logvar_, z_, None)
                # print("contrastive loss: ", loss)
                
                # -- EWC loss
                loss            = self.compute_ewc_penalty(loss)  # TODO: SA paper only uses one FIM, recompute FIM for each new task? right now, we use separate structs for each task and iterate    
                # print("EWC loss: ", loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) # update weights
    
            t2 = time.time()
            delta = (t2 - t1) * 1000.  # ms
            
            if i % 5 == 0: log.info(f'[TRAIN]\tstep\t{i}:\t' + f'loss: {loss.numpy()}, delta: {delta}')
            
            # self.dgr_model.metrics[0].update_state(loss)
            # self.dgr_model.metrics[1].update_state(delta)


    def init_fim_struct(self):
        for fim_var in self.fim_acc: fim_var.assign(fim_var * 0.)  # reset accumulators


    def compute_ewc_penalty(self, l):
        # for task_id in range(1, self.num_tasks):  # calculates the EWC loss penalty
        #     for var, var_prev, fim_var_prev in zip(self.trainable_variables, self.ewc_storage[task_id], self.fims[task_id]):
        #         l = l + self.coeffs[task_id] * tf.reduce_sum(fim_var_prev * (var - var_prev)**2)
        for var, var_prev, fim_var_prev in zip(self.trainable_variables, self.ewc_storage, self.fim):
            l = l + tf.reduce_sum(fim_var_prev * (var - var_prev)**2)
        return l * self.lambda_


    def compute_fim(self, fim_samples, past_classes):
        log.info(f'computing FIM for {fim_samples} samples.')
        for _ in range(fim_samples):
            # ---- use VAE generator to generate a random sample from classes to keep
            gen_x, gen_y = self.sample(eps=None, batch_size=1, scalar_classes=past_classes)
            
            with tf.GradientTape() as tape:
                mean, logvar    = self.encode(gen_x, gen_y)
                z               = self.reparameterize(mean, logvar)
                recon_x         = self.decode(z, gen_y)
                loss            = self.compute_loss(recon_x, gen_x, mean, logvar, z, None)

            gradients = tape.gradient(loss, self.trainable_variables)
                                      
            for grad, acc in zip(gradients, self.fim_acc): acc.assign(acc + grad**2)
    
    
    def finalize_fim(self, num_batches, batch_size):
        for i, fim_var in enumerate(self.fim_acc):
            fim_var.assign(fim_var / num_batches / batch_size)
            # print("i: ", i, fim_var.shape, fim_var.numpy().min(), fim_var.numpy().max())
            
        # update variables
        # for fim, acc in zip(self.fims[self.current_task], self.fim_acc): fim.assign(acc)
        # for var, storage in zip(self.trainable_variables, self.ewc_storage[self.current_task]): storage.assign(var)
        # self.coeffs[self.current_task].assign(tf.constant(1.0))
        for fim, acc in zip(self.fim, self.fim_acc): self.fim.assign(acc)
        for var, var_store in zip(self.trainable_variables, self.ewc_storage): self.ewc_storage.assign(var)
