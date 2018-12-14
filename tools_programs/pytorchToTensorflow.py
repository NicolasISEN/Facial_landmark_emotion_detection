import onnx
from keras.models import load_model

pytorch_model = '/path/to/pytorch/model'
keras_output = '/path/to/converted/keras/model.hdf5'
onnx.convert(pytorch_model, keras_output)