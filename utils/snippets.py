def tensorboard():
    print("""
    %load_ext tensorboard
    %tensorboard --logdir logs/ --port 6006

    Now open: http://localhost:6006
    """)

def imports():
    print("""
    # Add directory to path
    import sys; sys.path.insert(0, '/home/deeplearning/code/recursos')
    """)


def multi_metrics():
    print("""
    METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      f1_metric
]
    """)


def common_callbacks():
    print("""
    from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard

    model_checkpoint = ModelCheckpoint('model_name.h5', verbose = 1, save_freq = 'epoch')
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch')

    callbacks_list = [model_checkpoint, tensorboard]
    """)
