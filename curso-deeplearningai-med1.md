# CURSO 1 MEDICAL IMAGING DIAGNOSIS

* Aprender el uso del ground truth para etiquetar dataset
* Aprender a usar datos balanceados
* Aprender a hacer subtareas en histologia (con imagenes de alta resolucion, hacerlas pequeñas, haciendo parches)


Medical challenges:

* Class imbalance --> weighted loss / resampling
* Multi-task --> multi-label loss
* Dataset size --> transfer learning + data augmentation



EDAs:
```python
df.info() # tipos de variables
df.shape() # filas y columnas

# contar que los pacientes son unicos
train_df['PatientId'].count() #total
train_df['PatientId'].value_counts().shape[0] #unicos

train_df.columns # == train_df.keys()

# print labels
for column in columns:
    print(f"The class {column} has {train_df[column].sum()} samples")
```

With images we can get info like this:

```python
print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")
```


Generally the output is **prediction probabilities** for each label.
Loss function compares the error between the label and the prediction probability.



CLASS IMBALANCE

Dado que la mayor parte de la contribucion al loss total procede de ejemplos normales, esto provoca que la clasificación no se haga correctamente.

Por tanto, al binary loss se le aplica un peso (para ponderar los ejemplos positivos y negativos), y así que el loss total contribuya igual para los ejemplos positivos que para los negativos.

$$loss_{neg}^{(i)} = -1 \times weight_{neg}^{(i)} \times (1- y^{(i)}) \times log(1 - \hat{y}^{(i)} + \epsilon)$$

Se le añade epsilon para evitar que haga log de 0.
$$\epsilon = \text{a tiny positive number}$$

Por eso se le llama weighted loss.

Otro procedimiento es RESAMPLING que se hace en el dataset para conseguir que haya el mismo numero de normales y anormales.

Se puede hacer mediante oversampling de la clase inferior o undersampling de la clase superior (que suele significar dejar sin utilizar ejemplos normales).


MULTI-TASK CHALLENGE

Consiste en que queremos predecir varias enfermedades o problemas a la vez; por tanto necesitariamos varios modelos.

Pero podemos hacerlo solo de una vez y ademas eso es bueno porque las caracteristicas se comparten y es más rápido.

Para ello tenemos que modificar el loss de binary a multi-task setting:  

El multi-label loss sería el loss total que consiste en sumar los componentes indviduales del loss (de cada patologia por ej). 
De ahí simplemente tenemos que ponderar el loss para cada output utilizando el factor de weighted que le corresponde segun lo que hemos visto en el apartado anterior.

DATASET SIZE

Muchas veces no tenemos muchos datos (no tenemos millones como hace falta, sino 10000-100.000 ejempls). soluciones:

* transfer learning: entrenar para otra tarea y así aplicar su conocimiento en otra red médica (pretraining --> fine tuning)
    * early layers = general features
    * later layers = higher-level features

* generate more training samples = data augmentation
    * aplicar rotacion, zoom, cambio brillo, ...
    * deben REFLEJAR variaciones que existan en la vida real
    * deben PRESERVAR las labels (no provocar que cambie la patología)
    * ej: rotate-flip para derma, rotate+crop+color noise para histopato


SETS

* training data se separa entre
    * training set ) development of models
    * validation set = tuning and selection of models

a veces se hace cross-validation entre ellos

* test set = reporting of results


3 challenges in medicine:

* patient overlap = we have to make each set independent
* set sampling
* ground truth 


PATIENT OVERLAP
a veces memorizan caracteristicas de un paciente que esta en dos sets distintos y esto da lugar a una sobreestimación de la calidad del test set.

Forma parte del data leakage.

Pasos:
1. Extract patient IDs from the train and validation sets
2. Convert these arrays of numbers into `set()` datatypes for easy comparison
3. Identify patient overlap in the intersection of the two sets


Una forma de evitar este problema es no split per image (ya que asi aparecen imagenees del mismo paciente), sino usar split by patient, ya que solo aparece un paciente por set y por tanto no hay problema.


SET SAMPLING

al haber pocos casos, puede que al hacer la particion en test set haya pocos samples de patologia.

Por ello se suele poner que debe haber un % minimo de la clase minoritaria (a veces se pone un 50%).

generalmente primero se samplea el test set, luego el validation (que debe tener una distribución igual al test set)

por ultimo ya se samplea el train set


GROUND TRUTH (REFERENCE STANDARD)

Generalmente en medicina hay **consensus voting**, pero se puede hacer tanto como una sola voz como varias.

**More definitive test:** Tambien se puede usar un test mejor (confirmatorio), como el tc con la rx simple.