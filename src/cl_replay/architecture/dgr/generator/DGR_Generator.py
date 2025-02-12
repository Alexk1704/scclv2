import numpy as np
import tensorflow as tf

from cl_replay.api.utils import log
from cl_replay.api.data.Dataset import visualize_data


class DGR_Generator:


    def __init__(self, model, data_dims):
        self.model = model
        self.h, self.w, self.c, self.num_classes = data_dims


    def generate_data(self, task=-1, xs=None, gen_classes=None, stg=1000, sbs=100, vis_gen='no'):
        ''' generates samples from the old scholar (generator) '''
        log.debug('{:11s}'.format(' [GENERATOR] ').center(64, '~'))
        
        if self.model.generator_type == 'VAE':
            log.debug(f'gen_classes: {gen_classes}, enc. cond. input: {self.model.enc_cond_input}, dec. cond. input: {self.model.dec_cond_input}')

        generated_data = []
        generated_labels = []

        for iteration in range(0, int(stg) // int(sbs)):
            labels = None
            if self.model.generator_type == 'GAN':
                if self.model.conditional == 'yes':
                    labels = gen_classes
                    rnd_ys = np.random.choice(labels, size=sbs)
                    tmp = tf.eye(sbs, self.num_classes)
                    gen_ys = tf.gather(tmp, rnd_ys)
                else: gen_ys = None
                gen_xs = self.model.generator.sample(sbs, gen_ys)
                gen_xs = tf.reshape(gen_xs, (sbs, self.h, self.w, self.c))
                tmp_ys = self.model.solver.predict(gen_xs, verbose=0)
                amaxes = tmp_ys.argmax(axis=1)
                gen_ys = np.zeros(tmp_ys.shape)
                gen_ys[range(0,sbs), amaxes] = 1
            
            elif self.model.generator_type == 'VAE':
                if self.model.enc_cond_input == 'yes' or self.model.dec_cond_input == 'yes':
                    labels = gen_classes
                gen_xs, gen_ys = self.model.generator.sample(eps=None, batch_size=sbs, scalar_classes=labels)
            
            # NOTE: solver label prediction is turned off for CVAE since we generate the labels ourselves.
            if self.model.generator_type == 'VAE':
                if self.model.enc_cond_input == 'no' or self.model.dec_cond_input == 'no':
                    tmp_ys = self.model.solver.predict(gen_xs, verbose=0)
                    amaxes = tmp_ys.argmax(axis=1)
                    gen_xs = gen_xs.numpy()
                    gen_ys = np.zeros(tmp_ys.shape)
                    gen_ys[range(0,sbs), amaxes] = 1
            
            # if vis_gen == 'yes' and iteration == 0:
            #     xs_ = gen_xs.numpy()
            #     if not isinstance(gen_ys, np.ndarray): ys_ = gen_ys.numpy()
            #     else: ys_ = gen_ys
            #     visualize_data(xs_, ys_, self.model.vis_path, f'gen_T{task}')
            
            generated_data.append(gen_xs)
            generated_labels.append(gen_ys)
        concat_data = np.concatenate(generated_data, axis=0)
        concat_labels = np.concatenate(generated_labels, axis=0)
        
        #unique, counts = np.unique(concat_labels, return_counts=True)
        log.debug('{:8s}'.format(' [LABELS] ').center(64, '~'))
        log.debug(f'generated samples (classes) complete ds: {concat_labels.sum()}, {concat_labels.sum(axis=0)}')
        log.debug('{:5s}'.format('  ').center(64, '~'))
        
        return concat_data, concat_labels
