# KERAS
## Wandb

Primero instalar:

```python
# 1 - Instalar
!pip install wandb -q
!pip install opencv-python
import wandb
from wandb.keras import WandbCallback
```

Luego debemos crear los hiperparámetros:

* `wandb.init()` – Initialize a new W&B run. Each run is single execution of the training script.
* `wandb.config` – Save all your hyperparameters in a config object. This lets you use W&B app to sort and compare your runs by hyperparameter values.

```python
# Initilize a new wandb run
wandb.init(entity="santibacat", project="keras-intro", config=defaults)
# documentation https://docs.wandb.com/library/init
# config = dictionary with configuration, see https://docs.wandb.com/library/config

# Default values for hyper-parameters
config = wandb.config # Config is a variable that holds and saves hyperparameters and inputs
config.learning_rate = 0.01
config.epochs = 1000
config.img_width=28
config.img_height=28
config.num_classes = 10
config.batch_size = 128
config.validation_size = 5000
config.weight_decay = 0.0005
config.activation = 'relu'
config.optimizer = 'nadam'
config.seed = 42
```

Ahora creamos la red y debemos usar:

* El callback `WandbCallback()`, que debemos pasar a la red
* El magic `%%wandb` en cada celda que vaya a entrenar.

```python
%%wandb
# Fit the model to the training data
model.fit_generator(datagen.flow(X_train, y_train, batch_size=config.batch_size),
                    steps_per_epoch=len(X_train) / 32, epochs=config.epochs,
                    validation_data=(X_test, y_test), verbose=0,
                    callbacks=[WandbCallback(data_type="image", validation_data=(X_test, y_test), labels=character_names),
                                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
```

Para guardar **custom objects** debemos usar `wandb.log()`. Pueden ser imágenes, videos, HTML, plots...

```python
# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)


```
