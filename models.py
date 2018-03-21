from tensorflow.python.keras._impl.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.python.keras._impl.keras.layers import ZeroPadding2D as Padding, Cropping2D as Cropping
from tensorflow.python.keras._impl.keras.layers import Conv2D as Conv, Conv2DTranspose as Deconv
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D as Pooling, UpSampling2D as Sampling
from tensorflow.python.keras._impl.keras.models import Model

input_shape = (28, 36, 1)
latent_node = 64

# Recognizer
Padding1 = Padding((3, 2))
Conv1 = Conv(32, (5, 5), activation='relu', kernel_initializer='glorot_normal')
Pooling1 = Pooling((3, 3))

Padding2 = Padding((3, 2))
Conv2 = Conv(64, (5, 5), activation='relu', kernel_initializer='glorot_normal')
Pooling2 = Pooling((3, 3))

Padding3 = Padding((1, 1))
Conv3 = Conv(128, (3, 3), activation='relu', kernel_initializer='glorot_normal')
Pooling3 = Pooling((2, 2))

Padding4 = Padding((1, 1))
Conv4 = Conv(256, (3, 3), activation='relu', kernel_initializer='glorot_normal')
Pooling4 = Pooling((2, 2))

Flatten = Flatten()

# MLP-Encoder
Hidden1 = Dense(128, activation='softplus', kernel_initializer='glorot_normal')
Hidden2 = Dense(128, activation='softplus', kernel_initializer='glorot_normal')
Latent = Dense(latent_node, name='latent', kernel_initializer='glorot_normal')

# Categorizer
Categorize = Dense(12, activation='softmax', kernel_initializer='glorot_normal')

# MLP-Decoder
Unhidden2 = Dense(128, activation='softplus', kernel_initializer='glorot_normal')
Unhidden1 = Dense(128, activation='softplus', kernel_initializer='glorot_normal')
Unlatent = Dense(256, kernel_initializer='glorot_normal')

# Generator
Reshape = Reshape((1, 1, 256))

Sampling4 = Sampling((2, 2))
Deconv4 = Deconv(128, (3, 3), activation='relu', kernel_initializer='glorot_normal')
Cropping4 = Cropping((1, 1))

Sampling3 = Sampling((2, 2))
Deconv3 = Deconv(64, (3, 3), activation='relu', kernel_initializer='glorot_normal')
Cropping3 = Cropping((1, 1))

Sampling2 = Sampling((3, 3))
Deconv2 = Deconv(32, (5, 5), activation='relu', kernel_initializer='glorot_normal')
Cropping2 = Cropping((3, 2))

Sampling1 = Sampling((3, 3))
Deconv1 = Deconv(1, (5, 5), activation='sigmoid', kernel_initializer='glorot_normal')
Cropping1 = Cropping((3, 2))

def _Recognizer(x):
    x = Padding1(x)
    x = Conv1(x)
    x = Pooling1(x)
    x = Padding2(x)
    x = Conv2(x)
    x = Pooling2(x)
    x = Padding3(x)
    x = Conv3(x)
    x = Pooling3(x)
    x = Padding4(x)
    x = Conv4(x)
    x = Pooling4(x)
    x = Flatten(x)
    return x

def _MLP_Encoder(x):
    x = Hidden1(x)
    x = Hidden2(x)
    x = Latent(x)
    return x

def _Categorizer(x):
    y = Categorize(x)
    return y

def _MLP_Decoder(x):
    x = Unhidden2(x)
    x = Unhidden1(x)
    x = Unlatent(x)
    return x

def _Generator(x):
    x = Reshape(x)
    x = Sampling4(x)
    x = Deconv4(x)
    x = Cropping4(x)
    x = Sampling3(x)
    x = Deconv3(x)
    x = Cropping3(x)
    x = Sampling2(x)
    x = Deconv2(x)
    x = Cropping2(x)
    x = Sampling1(x)
    x = Deconv1(x)
    x = Cropping1(x)
    return x

x_input = Input(shape=input_shape)

x_conv_encoded = _Recognizer(x_input)
ConvEncoder = Model(x_input, x_conv_encoded)

x_conv_decoded = _Generator(x_conv_encoded)
ConvReconster = Model(x_input, x_conv_decoded)

x_mlp_encoded = _MLP_Encoder(x_conv_encoded)
Encoder = Model(x_input, x_mlp_encoded)

x_predicted = _Categorizer(x_mlp_encoded)
Categorizer = Model(x_input, x_predicted)

x_mlp_decoded = _MLP_Decoder(x_mlp_encoded)
x_reconsted = _Generator(x_mlp_decoded)
Reconstructor = Model(x_input, x_reconsted)
