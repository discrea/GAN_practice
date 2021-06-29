import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.python.compiler.mlcompute import mlcompute
"""
Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
위 오류의 발생 원인
    - Scikit-learn 내부에서 openblas를 사용한다.
    - openblas가 여러 개의 쓰레드를 사용하는데, 여러 라이브러리와의 충돌 원인이 된다.
    - numpy의 경우 완전히 이상한 값을 연산 결과로 내놓기도 한다.
해결 방법은 아래 코드 세줄
본 코드를 통해 openblas가 사용하는 Thread의 수를 제한함으로써 다른 라이브러리와의 충돌을 막을 수 있다.
"""
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ml_device = 'gpu'
out_dir = '../output_img/CNN_GAN'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
img_shape = (28,28,1)
epoch = 1000
batch_size = 128
noise_size = 100
sample_interval = 10

mlcompute.set_mlc_device(device_name=ml_device)

(X_train, _), (_, _) = mnist.load_data()
print('X_train.shape :', X_train.shape)

X_train = X_train / 127.5 - 1               # 255/2  => -1 ~ 1 사이의 값
X_train = np.expand_dims(X_train, axis=3)   # 차원을 하나 늘림. 3번 축으로
print('X_train.shape after np.expand_dims(X_train, axis=3) :', X_train.shape)

# build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise_size))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Reshape((7, 7, 256)))
generator_model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation="tanh"))
print(generator_model.summary())

# build discriminator => 여기서 mnist set이 들어감
discriminator_model = Sequential()
discriminator_model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['acc'])
discriminator_model.trainable = False

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())

gan_model.compile(loss='binary_crossentropy',
                  optimizer='adam')

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
    # discriminator_model.trainable = False
    # # False를 주는 이유 => fake값은 discriminaor, generator에 들어갈 때 두번 학습된다면
    # #                   real값은 경쟁이 안된다(학습량이 절반이 돼서)

    # CNN이 강력하기 떄문에 discriminator가 generator보다 훨씬학습이 잘되서
    # fake를 만드는 족족 다 잡아내게 된다. 그래서 generator 학습량을 N배로 주면서 서로 학습시킬 수 있게 만든다.
    # for i in range(10):
    z = np.random.normal(0, 1, (batch_size, noise_size))
    gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0:
        print('%4d | [D loss : %f, acc : %.2f%%] [G loss : %f]'%(
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