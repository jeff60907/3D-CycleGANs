import scipy
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D, UpSampling3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import read_tools
import numpy as np
import os


resolution = 64
batch_size = 1
###############################################################
config={}

config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = '../Data/'+name+'/train_25d/'
    config['Y_train_'+name] = '../Data/'+name+'/train_3d/'


config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = '../Data/'+name+'/train_25d/'
    config['Y_test_'+name] = '../Data/'+name+'/train_3d/'

config['resolution'] = resolution
config['batch_size'] = batch_size

################################################################

DA_loss = []
DB_loss = []
G_loss = []

if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
if not os.path.exists('images'):
    os.makedirs('images')


# Plot the loss from each batch
def plotLoss(name, losses, Label):
    plt.figure(figsize=(10, 8))
    plt.plot(losses, label=Label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/' + name + '.png')
    plt.close()


class CycleGAN():
    def __init__(self):
        # Input data shape
	self.cube_len = 64
	self.channels = 1
	self.data_shape = (self.cube_len, self.cube_len, self.cube_len, self.channels) # 64x64x64x1


        # Calculate output shape of D (PatchGAN)
        patch = int(self.cube_len / 2**4)
        self.disc_patch = (patch, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Loss weights
        self.lambda_cycle = 5.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
	self.d_A.summary()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct 3D models of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
	self.g_AB.summary()

        # Input sparse/dense from both domains
        data_A = Input(shape=self.data_shape)
        data_B = Input(shape=self.data_shape)

        # Translate 3D models to the other domain
        fake_B = self.g_AB(data_A)
        fake_A = self.g_BA(data_B)
        # Translate 3D models back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of 3D models
        data_A_id = self.g_BA(data_A)
        data_B_id = self.g_AB(data_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[data_A, data_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        data_A_id, data_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mse', 'mse',
                                    'mse', 'mse'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
	self.g_AB.compile(loss='mse',optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv3d(layer_input, filters, f_size=(4,4,4)):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=(4,4,4), dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=(2, 2, 2))(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=(1,1,1), padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u
        # 3D models input
        d0 = Input(shape=self.data_shape)

        # Downsampling
        d1 = conv3d(d0, self.gf)
        d2 = conv3d(d1, self.gf*2)
        d3 = conv3d(d2, self.gf*4)
        d4 = conv3d(d3, self.gf*8)

        # Upsampling
        u1 = deconv3d(d4, d3, self.gf*4)
        u2 = deconv3d(u1, d2, self.gf*2)
        u3 = deconv3d(u2, d1, self.gf)
 
	u4 = UpSampling3D(size=(2,2,2))(u3)
        output_data = Conv3D(self.channels, kernel_size=(4,4,4), strides=(1,1,1), padding='same', activation='sigmoid')(u4)

        return Model(d0, output_data)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=(1,1,1), normalization=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=(2,2,2), padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        data = Input(shape=self.data_shape)

        d1 = d_layer(data, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv3D(1, kernel_size=(4,4,4), strides=(1,1,1), padding='same')(d4)

        return Model(data, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
	    Data.shuffle_X_Y_files(label='train')
	    total_num = len(Data.X_train_files)
	    for batch_i in range(total_num//batch_size):
	        print ('-'*3, 'batch_num %d / %d' % (batch_i, total_num//batch_size), '-'*3)
		data_A, data_B = Data.load_X_Y_voxel_grids_train_next_batch()
		data_B = data_B/2
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate 3d models to opposite domain
                fake_B = self.g_AB.predict(data_A)
                fake_A = self.g_BA.predict(data_B)

                # Train the discriminators (original models = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(data_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(data_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Extra train the sparse2dense generator
		au_loss = self.g_AB.train_on_batch(data_A, data_B)

		# Train combined cycleGANs
                g_loss = self.combined.train_on_batch([data_A, data_B],
                                                        [valid, valid,
                                                        data_A, data_B,
                                                        data_A, data_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, total_num//batch_size,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))	
		G_loss.append(g_loss[0])
		DA_loss.append(d_loss[0])
		DB_loss.append(dB_loss[0])

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
		    self.save_data(epoch, batch_i, data_A, data_B)
		    self.g_AB.save('./saved_model/g_AB_'+str(epoch)+'.h5')
		    self.g_BA.save('./saved_model/g_BA_'+str(epoch)+'.h5')
		    self.d_B.save('./saved_model/d_B_'+str(epoch)+'.h5')
		    self.d_A.save('./saved_model/d_A_'+str(epoch)+'.h5')
	            np.save('saved_model/G_Loss',G_loss)
	    	    np.save('saved_model/D_Loss',DA_loss)
	    	    np.save('saved_model/DA_Loss',DB_loss)
	            plotLoss(name='G_loss', losses=G_loss, Label='Generator loss')
		    plotLoss(name='DA_loss', losses=DA_loss, Label='Discriminitive_A loss')
		    plotLoss(name='DB_loss', losses=DB_loss, Label='Discriminitive_B loss')

    def save_data(self, epoch, batch_i,x,y):
        fake_B = self.g_AB.predict(x)
        fake_A = self.g_BA.predict(y)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'_25d'+Data.batch_name[0][29:-13] , x[0])
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'_real'+Data.batch_name[0][29:-13] , y[0])
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'fake_B'+Data.batch_name[0][29:-13] , fake_B[0])
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'fake_A'+Data.batch_name[0][29:-13] , fake_A[0])
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'reconstr_A'+Data.batch_name[0][29:-13] , reconstr_A[0])
	Data.plotFromVoxels(str(epoch) +'_'+str(batch_i)+'reconstr_B'+Data.batch_name[0][29:-13] , reconstr_B[0])
	Data.output_Voxels(str(epoch) +'_'+str(batch_i)+'_fake'+Data.batch_name[0][29:-13], fake_B[0])


if __name__ == '__main__':
    Data = read_tools.Data(config)
    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=20)
