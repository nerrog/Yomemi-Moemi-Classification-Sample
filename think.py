# -*- coding: utf-8 -*-
# 引数に指定された画像を推測するスクリプト

import keras
import sys
import os
import numpy as np
from keras.models import load_model
from PIL import Image

imsize = (64, 64)


# モデルファイルの指定、判別するファイルは実行時の引数から取得
name = sys.argv[1]
testpic = name
keras_param = "./cnn.h5"


def load_image(path):
    # 画像サイズを都合の良いサイズに変換し、numpy用に変換する
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    img = np.asarray(img)
    img = img / 255.0
    return img


model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))

print(prd)  # 精度を表示
prelabel = np.argmax(prd, axis=1)
if prelabel == 0:
    print(">>> 萌実ちゃん")
elif prelabel == 1:
    print(">>> ヨメミちゃん")
