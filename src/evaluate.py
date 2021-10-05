# %%
# ライブラリのインポート
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from natsort import natsorted
from tensorflow.keras.models import load_model


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


# %%
# ルートディレクトリの取得
rootdir = Path(__file__).parent.parent

# %%
# 学習モデルのロード
model = load_model(rootdir.joinpath("model.h5"))


# テストデータの読み込み
test_path = natsorted(list(rootdir.joinpath("Data/test/").glob("*.png")), key=lambda x: x.name)
test_path = [str(test_path[i]) for i in range(len(test_path))]
test_img_list = read_img(test_path)

test_label_list = pd.read_csv(rootdir.joinpath("Data/test/labels.csv"), header=None)
test_label_list = test_label_list[0].to_list()
test_img_ds, test_label_ds = create_dataset(test_img_list, test_label_list)

print("----------------evaluate_model----------------")
test_loss, test_acc = model.evaluate(test_img_ds, test_label_ds)
print(" ")
print("result: test_acc = " + str(test_acc))

x = {
    "test_acc": test_acc
}

print(x)
with open(Path(__file__).parent.parent.joinpath("scores.json"), 'w') as f:
    json.dump(x, f, ensure_ascii=False, indent=4)
