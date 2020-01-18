# ライブラリインポート
import os.path, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../../')
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
from keras.optimizers import SGD


N_CATEGORIES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

pwd = os.getcwd()
print(pwd)

# 画像ディレクトリのパス
train_dir = 'images/train/'
validation_dir = 'images/validation/'

# 学習データサイズ
NUM_TRAINING = 1600
NUM_VALIDATION = 400

# 入力画像情報(IMAGE_SIZE*IMAGE_SIZEのRGB)
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# 追加する層を定義(最終層)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print()

# 訓練データ画像の水増し
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)
# 水増し
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    # shuffle=True
)
validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    # shuffle=True
)

hist = model.fit_generator(train_generator,
                           steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
                           epochs=EPOCHS,
                           verbose=1,
                           validation_data=validation_generator,
                           validation_steps=NUM_VALIDATION//BATCH_SIZE,
                           )

model.save('model-1.h5')

