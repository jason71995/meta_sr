from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error
from model.meta_edsr import build_model
from utils.common import data_loader

batch_size = 16
epochs = 200
train_steps = 1000
val_steps = 100
patch_size = 50
max_scale = 4.0
train_data_path = "DIV2K/train"
val_data_path = "DIV2K/val"

train_data_loader = data_loader(train_data_path, max_scale, batch_size, patch_size, augmentation=True, preload_all_image=False)
val_data_loader   = data_loader(val_data_path,   max_scale, batch_size, patch_size, augmentation=True, preload_all_image=False)

lr_decay = 9.0 / (epochs * train_steps)

model = build_model(output_channel=3, filters=64, block=16)
model.compile(
    optimizer=Adam(lr=1e-4, decay=lr_decay),
    loss=mean_absolute_error
)
model.fit_generator(
    train_data_loader,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_data=val_data_loader,
    validation_steps=val_steps,
    callbacks=[
        ModelCheckpoint(
            'weights.h5',
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min"),
    ]
)