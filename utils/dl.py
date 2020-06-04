def plot_history(history, half_epoch=False):
    """
    Plots train-val metrics from keras model.history
    Graph 1: Losses
    Graph 2: Accuracy
    Usage: plot_history(model.history)
    """
    import matplotlib.pyplot as plt

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[val_loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    if half_epoch: # por si se para a mitad, el val_loss y el train_loss son distintos
        for l in loss_list:
            plt.plot(epochs, history.history[l][:-1], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    else:
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    if half_epoch:
        for l in acc_list:
            plt.plot(epochs, history.history[l][:-1], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    else:
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    

    
def print_metrics(y_test, y_pred):
    """Prints and returns metrics for y_test, y_pred keras pair"""
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    return precision, recall, f1




def image_grid_from_dfcolumn(df, filter, column_path = 'Path', num_samples=9):
    """Taken from dl.ai med 1 diagnosis"""
    imgs = [df.loc[filter,column_path].sample().values for i in range(num_samples)]
    
#     print(f'Printing df {df} with filter [{filter}]') NO SE HACERLO
    # Adjust the size of your images
    plt.figure(figsize=(20,10))

    # Iterate and plot random images
    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(imgs[i][0]) # es como una lista anidada y no se cambiarlo
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()  