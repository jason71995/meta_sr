from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Input,Add
from utils.layer import MetaUpSample

def res_block(x, filters):
    y = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    y = Conv2D(filters, (3, 3), padding="same")(y)
    y = Add()([y, x])
    return y

def build_model(output_channel, filters, block, meta_ksize = (3,3)):

    image = Input((None,None,output_channel))
    res = Conv2D(filters,(1,1), padding="same")(image)
    y = res
    for _ in range(block):
      y = res_block(y,filters)
    y = Add()([y,res])

    coord = Input((None,None,3))
    meta_w = Dense(256, activation="relu")(coord)
    meta_w = Dense(meta_ksize[0] * meta_ksize[1] * filters * output_channel)(meta_w)

    y = MetaUpSample(output_channel, meta_ksize)([y, meta_w])
    return Model([image, coord], [y])