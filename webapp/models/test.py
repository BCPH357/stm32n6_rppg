import tensorflow as tf

# 1. 建立一個最小模型
inputs = tf.keras.Input(shape=(36, 36, 3), name="input_0")
x = tf.keras.layers.Conv2D(4, (3, 3), activation="relu")(inputs)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(1, activation=None, name="output_0")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# 2. 建立 TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 設定 float32 模式（最簡單）
converter.optimizations = []
tflite_model = converter.convert()

# 4. 儲存成 tflite
with open("test_n6_ok.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved test_n6_ok.tflite")
