def create_fullpath(df):
    """Para crear fullpath desde la ruta raiz y el df"""
    import os
    df['Path'] = df.apply(lambda x: os.path.join('/media/DATOS/ORIG/padchest', x['ImageDir'], x['ImageID']), axis=1)
    return df

def print_img(img, path=None):
    import matplotlib.pyplot as plt
    import matplotlib, os
    if path:
        plt.imshow(matplotlib.image.imread(os.path.join(path, img), cmap='gray'))
    else:
        plt.imshow(matplotlib.image.imread(os.path.join(img), cmap='gray'))

