# %%
# ライブラリのインポート
import numpy as np
import pandas as pd
import cv2
import yaml
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from natsort import natsorted


# %%
# 画像読み込み関数
def read_img(pathlist: list) -> list:
    data_list = []
    for path in pathlist:
        with open(path):
            data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            data = data[:, :, np.newaxis]
        data_list.append(data)
    return data_list


# データセットの作成
def create_dataset(img_list: list, label_list: list):
    img_ds = np.array(img_list)
    img_ds = img_ds.astype("float32")
    img_ds /= 255.0
    label_ds = np.array(label_list)
    label_ds = to_categorical(label_ds.astype('int32'), 10)
    return img_ds, label_ds


# モデルの作成
def create_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# %%
# ルートディレクトリの取得
rootdir = Path(__file__).parent.parent

# %%
# params.yamlからパラメーターの読み込み
params = yaml.safe_load(open(rootdir.joinpath("params.yaml")))["train"]
split = params["split"]
batchsize = params["batchsize"]
epoch = params["epoch"]
loss = params["loss"]
optimizer = params["optimizer"]

# %%
# training,validation,testデータのパス取得、strに変換
train_path = natsorted(list(rootdir.joinpath("Data/train/").glob("*.png")), key=lambda x: x.name)
train_path = [str(train_path[i]) for i in range(len(train_path))]
train_img_list = read_img(train_path)

valid_path = natsorted(list(rootdir.joinpath("Data/valid/").glob("*.png")), key=lambda x: x.name)
valid_path = [str(valid_path[i]) for i in range(len(valid_path))]
valid_img_list = read_img(valid_path)


# training,validation,testのラベルの取得
train_label_list = pd.read_csv(rootdir.joinpath("Data/train/labels.csv"), header=None)
train_label_list = train_label_list[0].to_list()

valid_label_list = pd.read_csv(rootdir.joinpath("Data/valid/labels.csv"), header=None)
valid_label_list = valid_label_list[0].to_list()

# tensorflow学習用のデータセットに変換
train_img_ds, train_label_ds = create_dataset(train_img_list, train_label_list)
valid_img_ds, valid_label_ds = create_dataset(valid_img_list, valid_label_list)


# %%
# モデルの作成
model = create_model()

# %%
# 学習スタート
print("-------------------start-------------------")
hist = model.fit(train_img_ds, train_label_ds, batch_size=batchsize, epochs=epoch, verbose=1, validation_data=(valid_img_ds, valid_label_ds))
print("-------------------finish-------------------")
# %%
# モデルの保存
model.save(rootdir.joinpath("model.h5"))
