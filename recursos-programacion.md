# WORKFLOW

## EDA

* Hacer tSNE para hacernos una idea de la variabilidad del modelo
* 
## MODEL

### Preprocessing

* Normalization = 0-255
* Contrast stretching

### Imbalanced data

* ADASYN = generar datos sinteticos para clases con menos ejemplos
* SMOTE = oversampling clases minoritarias y undersampling mayoritarias --> Data augmentation


### Ensemble

* Ensemble = agrupar prediciones de varios modelos (computationally expensive).
* Snapshot ensembling = guardar parametros del modelo periodicamente y usar cada uno como un 'minimodelo' ya que tienen distintos local minima.

Anti-aliasing = prevenir shift-invariant, para ello añadir capa BlurPool.


## RESULTS





# PACKAGES

## PAQUETES INTERESANTES 

https://towardsdatascience.com/python-tools-for-a-beginner-data-scientist-39b3b9a4303a

**Beautiful soup**  
:	Descargar texto html o xml

**Wget**  
:	descargar archivos. 
	Uso: `wget.download('http://:..")`

**Pendulum**  
: Util para manejo de tiempo en python

**Barras de progreso:**  
:	tqdm

**Conocer tiempo de ejecución:**  
:	timebudget

**servidor HTTP python**:  
:	`python -m http.server [<portNo>]`


## DATA SCIENCE

* Ejecutar ml y data science en el navegador
Streamlit

* Data labeling:
  * Label studio

## MACHINE LEARNING

StatsModels  
	Liberia de estadistica en python

Librerias de machine learning 

* Scikit-learn
* XGBoost, LightGBM, Catboots
* Eli5
* Para NLP: NLTK, SpaCy, Gensim
* Data scraping. Scrapy
Imbalanced-learn
                Tomek-links

## DEEP LEARNING

wandb para ver como va el trainig (tipo tensorboard)

# CODE SNIPPETS
https://snippets.readthedocs.io/en/latest/index.html

## Añadir utils
```python
import sys; sys.path.insert(0, '/home/deeplearning/code/recursos')
import utils
utils.basic.test_gpu()
```




---

# UTILIDADES

## DEEP LEARNING

* [Plantilla proyectos tensorflow](https://github.com/Mrgemy95/Tensorflow-Project-Template)

https://snippets.readthedocs.io/en/latest/index.html