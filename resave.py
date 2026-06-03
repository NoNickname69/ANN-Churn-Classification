import tensorflow as tf 
import keras 
model = keras.models.load_model('C:/Users/goggl/ANN Example/model.keras') 
model.save('C:/Users/goggl/ANN Example/model_v2.h5') 
print('Done!') 
