from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
import numpy as np

classes = ["Moemi", "Yomemi"]
num_classes = len(classes)
image_size = 64


def load_data():
    # npy形式のデータセットを指定
    X_train, X_test, y_train, y_test = np.load(
        "./dataset.npy", allow_pickle=True)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

# train


def train(X, y, X_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = RMSprop(lr=0.00005, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    history = model.fit(X, y, batch_size=32, epochs=15)
    # HDF5ファイルにKerasのモデルを保存
    model.save('./cnn.h5')

    # 学習過程を可視化

    metrics = ['loss', 'accuracy']  # 評価関数を指定

    plt.figure(figsize=(10, 5))

    for i in range(len(metrics)):

        metric = metrics[i]

        plt.subplot(1, 2, i+1)
        plt.title(metric)

        plt_train = history.history[metric]

        plt.plot(plt_train, label='training')
        plt.legend()

        plt.show()  # グラフの表示

    # モデル構造の可視化
    model.summary()

    return model


def main():
    # データ読み込み
    X_train, y_train, X_test, y_test = load_data()

    # モデルの学習
    model = train(X_train, y_train, X_test, y_test)


main()
