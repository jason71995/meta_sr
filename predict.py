from model.meta_edsr import build_model
from utils.common import get_coords
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default=None, type=str, help="Image path.")
parser.add_argument("-m", "--model", default=None, type=str, help="Model path.")
parser.add_argument("-s", "--scale", default=None, type=float, help="Up scale factor.")
args = parser.parse_args()

assert args.scale>=1.0, "Scale factor must greater than 1.0"

model = build_model(output_channel=3,filters=64,block=16)
model.load_weights(args.model)

lr_image = Image.open(args.image).convert("RGB")
lr_w, lr_h = lr_image.size
hr_w, hr_h = int(lr_w * args.scale), int(lr_h * args.scale)

input_image = np.expand_dims(np.asarray(lr_image, "float32") / 127.5 - 1, axis=0)
input_coord = np.expand_dims(get_coords((hr_h, hr_w), (lr_h, lr_w), args.scale), 0)

pred_image = model.predict([input_image, input_coord], batch_size=1)

pred_image = Image.fromarray(np.clip((pred_image[0] + 1) * 127.5, 0, 255).astype("uint8"), "RGB")
pred_image.save("img_meta-sr_x{0}.png".format(args.scale))