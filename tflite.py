import tensorflow as tf

# Keras 모델 불러오기
model = tf.keras.models.load_model('my_model/model1.keras')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 입력 및 출력 형식 지정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

# 변환된 모델 저장
tflite_model = converter.convert()
tflite_model_path = 'my_model/model1.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

