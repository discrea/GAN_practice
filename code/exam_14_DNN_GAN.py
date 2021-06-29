import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.python.compiler.mlcompute import mlcompute

ml_device = 'gpu'
out_dir = '../output_img/'
img_shape = (28,28,1)
epoch = 100000
batch_size = 128
noise_size = 100
sample_interval = 100

mlcompute.set_mlc_device(device_name=ml_device)

(X_train, _), (_, _) = mnist.load_data()
print('X_train.shape :', X_train.shape)

X_train = X_train / 127.5 - 1               # 255/2  => -1 ~ 1 사이의 값
X_train = np.expand_dims(X_train, axis=3)   # 차원을 하나 늘림. 3번 축으로
print('X_train.shape after np.expand_dims(X_train, axis=3) :', X_train.shape)

# build generator => 100개짜리 잡음만 줄거임. mnist set이랑은 아무 관련 없음
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise_size))
# LeakyReLu => train 값이 0~1이 아니라 -1~1이어서 ReLu대신 씀(음수 부분에서 약간은 기울기를 가짐)
# LeakyReLu는 alpha 값(음의 영역대 기울기)을 줘야 해서 Dence 생성시 넣지 않는다
# 굳이 Dense안에 넣겠다면
# lrelu = LeakyReLu(alpha=0.01)
# generator_model.add(Dense(128, activation=lrelu, input_dim=100))
generator_model.add(LeakyReLU(alpha=0.01))      # LeakyReLu는 엑티베이션 펑션, 여기까지는 레이어 한층이다
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape))
print(generator_model.summary())

# build discriminator => 여기서 mnist set이 들어감
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['acc']
                            )
discriminator_model.trainable = False

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())

gan_model.compile(loss='binary_crossentropy',
                  optimizer='adam'
                  )

real = np.ones((batch_size,1))
print('real[:3] :\n', real[:3])
fake = np.zeros((batch_size,1))
print('fake[:3] :\n', fake[:3])

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise_size))
    fake_imgs = generator_model.predict((z))

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator_model.trainable = False
    # False를 주는 이유 => fake값은 discriminaor, generator에 들어갈 때 두번 학습된다면
    #                   real값은 경쟁이 안된다(학습량이 절반이 돼서)

    z = np.random.normal(0, 1, (batch_size, noise_size))
    gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0:
        print('%d [D loss : %f, acc : %.2f%%] [G loss : %f]'%(
            itr, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row*col, noise_size))
        fake_imgs = generator_model.predict((z))
        fake_imgs - 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        path = os.path.join(out_dir, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()

