from django.db import models
import numpy as np
import keras, sys
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io, base64

graph = tf.compat.v1.get_default_graph()  # tfのノード計算グラフ
class Photo(models.Model):
    image = models.ImageField(upload_to="photos")

    IMAGE_SIZE = 224  # 入力画像サイズ
    MODEL_PATH = "./classai/ml_models/model-1.h5"  # 学習済みモデル
    classes = ["アヒル", "ハト", "白鳥"]  # 教師ラベル
    image_len = len(classes)

    def predict(self):  # 推論・評価
        model=None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_PATH)

            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")  # カラー指定
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image)/255.0  # 正規化
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted]*100)

            return self.classes[predicted],percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return "data:"+img.file.content_type+";base64,"+base64_img