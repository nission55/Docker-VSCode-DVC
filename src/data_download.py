# %%
# ライブラリのインポート
import numpy as np
import cv2
from pathlib import Path
from tensorflow import keras

# %%
# fashion-mnistのデータセットのダウンロード
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %%
# 画像の保存
for i in range(50000):
    savepath = Path(__file__).parent.parent.joinpath("Data/train/", "{}.png".format(i))
    savepath = str(savepath)
    cv2.imwrite(savepath, train_images[i])

# %%
# ラベルの保存
train_labels = train_labels[0:50000]
train_labels = np.array(train_labels).astype("int")
np.savetxt(Path(__file__).parent.parent.joinpath("Data/train/labels.csv"), train_labels, delimiter=",", fmt="%.0f")
# %%
