
# PANDAS


https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428

## TL;DR
Gui for Pandas = Bamboolib
PARA QUE SE VEAN TODAS LAS FILAS O COLUMNAS
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500


## BASICS

`df.info`  
`df.dtypes` = ver tipos de cada columna  
`df.describe()` = te da media, etc de cada columna  
`df.groupby`  

```py
df.groupby(by=df['Incident Date'].dt.hour)
```

`len(df)` = da el nmero de filas (training samples)
`df.shape`

-----

Para aplicar una funcion a un dataset
```
def funcion(x):
df['columna'].apply(funcion)
```

Es lo mismo que

`df['columna'].apply(lambda x: 0. if '-' in str(x) else x)`


-----

Seleccionar una columna  
	`df['state']`

o tambien  
	`df.state` 
    
(lo unico que esto solo si no tiene espacios y no es igual que un nombre de un metodo. NO USAR)

**Tipos de datos**

* Index y columnas = pandas index object (es como una secuencia de etiquetas para columnas y filas)

* Values = numpy ndarray (df.as_matrix)
  * index = `df.index`
  * columns = `df.columns`
  * values = `df.values`

Seleccionar filas por etiqueta o por localizacion:
* `loc`: indeexa por etiquetas (labels)
* `iloc`: indexa por localizacion (numero de indice)

NO USAR AT e IAT
	
```
.loc
	df.loc[row_selection, column_selection]
	df.loc[['Dean', 'Cornelia'], ['age', 'state', 'score']] 
		# The .loc indexer simultaneously select subsets of rows and columns by the LABEL. 
		INCLUYE EL ULTIMO VALOR (no como un rango de python)
	>>> df.loc['Niko':'Dean'] (incluye de fila Niko hasta Dean)
	>>> df.loc['Niko':'Christina':2] # stepping by 2
```	

No recomendado usar `df[inicio:fin]` para elegir filas, mejor usar `.loc` o `.iloc`

> Using .iloc and .loc is explicit and clearly tells the person reading the code what is going to happen. Let's rewrite the above using .iloc and .loc.
	>> df.iloc[3:6]      # More explicit that df[3:6]
	
	
Seleccionar varios con doble corchete:
* `df['food']` > devuelve una Series
* `df[['food']] `>> devuelve un Dataframe
    * df[['color', 'food', 'score']]


`read_csv` y `read_table `es lo mismo, mejor usar read_csv
	read_csv (sep=, decimal=)

`isna-isnull and notna-notnull` nos dan un df con valores booleanos.
* Mejor usar isna (no notna)
> You can also avoid ever using notna since Pandas provides the inversion operator, ~ to invert boolean DataFrames.

`dropna() `nos elimina las filas NA
	sats = college_idx[['satmtmid', 'satvrmid']].dropna()

	

**Copiar un df**
Por defecto df2 = df1 si modificas algo en uno cambia en el otro
hay que hacer df2 = df1.copy()

`**map**`
sirve para hacer transformaciones de datos
You first define a dictionary with ‘keys’ being the old values and ‘values’ being the new values. 
	level_map = {1: 'high', 2: 'medium', 3: 'low'} df['c_level'] = df['c'].map(level_map) 

`**cut**`
Para crear bins (intervalos discretos)
Ej: pd.cut(df['Age'], 3, labels=['low','mid','high'])

`**pivot_table**`
crea tablas pivotadas tipo excel


COMBINAR DATASETS
[Ref](https://towardsdatascience.com/combining-pandas-dataframes-the-easy-way-41eb0f2c1ebf)

![](https://miro.medium.com/max/2000/1*B_0ZwHjGz2YMR9pqXtVljw.png)


Inner join: solo une lo que esté en los dos
Left/right: todo lo del izquierdo y la parte comun del derecho
Outer join: todo lo de los dos (aunque falte en el otro).


Modificar df desde jupyter notebook con qgrid
https://medium.com/@williams.evanpaul/edit-pandas-dataframes-within-jupyterlab-36f129129496

Panda avanzado:idxmax, idxmin() = devuelve la posicion (index) del minimo-maximo
ne(valor) = devuelve True si el valor es el que has puesto, sino False 
nsmallest, nlargest(numero, 'columna') = devuelve los x valores menores-mayores de una columna


A partir de pandas 1, se puede convertir una columna en markdown:
```python
pip install tabulate #requisito
df.to_markdown()
```



# NUMPY
`np.where()` = devuelve el indice de los elementos que cumplan una condicion:
    `np.where(grades >6)`

Tambien se puede usar para cambiar los que cumplen la condicion y los que no
    `np.where(grades>3,  'si', 'no)`

`argmin, argmax()`
devuelve los indices con los valores minimos o maximos

`argsort()`
devuelve los indices ordenados de menos a mayor

`intersect1d()`
crea una combinacion de elementos comune sde dos array (no el index sino el valor)

`allclose()`
comprueba si dos arrays son similares según un umbral de tolerancia
    `np.allclose(arr1,arr2,0.1)`

# ANACONDA

* Ver paquetes:  
    `conda list`

* Ver entornos:  
    `conda env list`
	
* Crear entorno  
	`conda create –n nuevoentorno python=3.7`  
	`conda activate nuevoentorno`

* Instalar paquetes  
	`conda install numpy=1.10`  
	`conda install -c some-channel packagename`
	
* Para añadir un nuevo canal  
	`conda config --add channels some-channel`
	
* Eliminar canal  
	`conda remove --name myenv --all`

* Actualizar todos los paquetes  
	`conda update conda && conda update anaconda`  
	`conda update --all`

* Limpiar paquetes no usados:  
	`conda clean -a`

* Para añadir conda-forge:  
    `conda config --add channels conda-forge`  
    `conda config --set channel_priority strict`  
    `conda install <package-name>`  

[Conda cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf):
![](https://i1.wp.com/data-forest.co/wp-content/uploads/2019/01/conda-cheatsheet-001.jpg?w=792&ssl=1)



# JUPYTER

* **jupyterhub** es para instalar para muchos usuarios a la vez (ej en una clase).
	* no lo voy a usar en principio
* **jupyterlab** es la evolución de jupyter notebook
	* si la voy a usar

## JUPYTER-NOTEBOOK

EXTENSIONES JUPYTER

`conda install -c conda-forge jupyter_nbextensions_configurator ipywidgets nodejs jupyterlab`

**MEJORES EXTENSIONES JUPYTER**
* Hinterland = autocompleta texto
* Splitcells = divide celdas verticalmente
* Snippets = menu para insertar codigo pequeño con explicacion
* Table of Contents = deja seleccionar contenido en el lateral
* Collapsible headings = permite colapsar headings (para pruebas etc)
* ExecuteTime = calcular tiempo de ejecucion

No usadas: 

* Variable inspector = te dice las variables que tienes activas
* Notify = manda una notificacion cuando acaba
* Codefolding = permite colapsar codigo si tienes mierda
* %debug = si escribimos esto despues de un error depuramos el codigo
* autopep8 = estiliza el codigo
* qgrid = pandas df interactivo en jupyter notebook


https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/
https://medium.com/swlh/some-neat-jupyter-tricks-be0775c3f17


Documentacion facil
`?str.replace()`

**Multicursor** = Arrastrar el mouse apretando alt para cambiar multiples cursres

**Exportar con otro formato un ipynb con nbconvert**
jupyter nbconvert --to my_format my_notebook.ipynb


**DEBUGGING**
https://twitter.com/radekosmulski/status/945739571735748609
Tras una excepción, en una nueva celda escribe %debug
Esto mostrará un depurador interactivo:
	• si escribimos help muestra superpoderes
	• up nos lleva a un nivel más alto
podemos integrarlo dentro de una funcion de esta forma:
```py
from IPython.core.debugger import set_trace
def full_speed_ahead_from_script(direction='N'):
    set_trace()
    pass
```

## JUPYTER-LAB

https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html
https://towardsdatascience.com/3-must-have-jupyterlab-2-0-extensions-41024fe455cc

**EXTENSIONES JUPYTERLAB**
* table of contents: `jupyter labextension install @jupyterlab/toc`
* github: `jupyter labextension install @jupyterlab/github`
* google-drive: `jupyter labextension install @jupyterlab/google-drive`
* git:  
```jupyter labextension install @jupyterlab/git
pip install jupyterlab-git
jupyter serverextension enable --py jupyterlab_git
```
* [qgrid](https://github.com/quantopian/qgrid) (abrir pandas interactivamente en notebook): `jupyter labextension install qgrid `
* abrir xls xlsx: `jupyter labextension install jupyterlab-spreadsheet`
* [variable inspector](https://github.com/lckr/jupyterlab-variableInspector): `jupyterlab-variableInspector `

OTRAS
* para hacer graficos y flowcharts `jupyter labextension install jupyterlab-drawio`
* para buscar archivos y abrirlos rapido `jupyterlab-quickopen `
* para abrir archivos latex `jupyterlab-latex `

Tambien se puede usar el comando uninstall, update, disable


Basico
 https://towardsdatascience.com/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b

Aprender en Jupyterlab a abrir varios archivos y compartir variables
https://towardsdatascience.com/jupyterlab-a-next-gen-python-data-science-ide-562d216b023d

**Usar otro kernel**
Use kernel from > la otra abierta

### UTILIDADES

* [nbviewer](https://nbviewer.jupyter.org/) visor ipynb online
* [nbconvert](https://nbconvert.readthedocs.io/en/latest/) convertir ipynb a html-pdf-etc
* [nbdev](https://nbdev.fast.ai/) para desarrollar una libreria en ipynb
* [Binder](https://mybinder.org/) turn Github repo into Jupyter Notebooks
* [Jupytext]() turn ipynb to md/py y viceversa automaticamente. Util para Git y control de versiones
* [Dataframe_image](https://github.com/dexplo/dataframe_image) para imprimir df como imagenes para que se quede bien en pdf (en vez de como texto)



[Data science workflow](https://www.fast.ai/2020/01/07/data-questionnaire/)

# VISUALIZATION

## MATPLOTLIB

Está en onenote

https://medium.com/@kapil.mathur1987/matplotlib-an-introduction-to-its-object-oriented-interface-a318b1530aed

Cheatsheet matplotlib
https://github.com/rougier/matplotlib-cheatsheet

Libro de visualizacion del mismo autor
https://github.com/rougier/scientific-visualization-book


## SEABORN 
https://medium.com/@neuralnets/statistical-data-visualization-series-with-python-and-seaborn-for-data-science-5a73b128851d
> Part 1 — Loading Datasets and Controlling Aesthetics
> Part 2 — More on Aesthetics and comparison with Matplotlib
> Part 3— Color Palettes
> Part 4— Regression Plots (LM Plot and Reg Plot)
> Part 5— Linear Data (Scatter plot and Joint Plots)
> Part 6— Additional Regression Plots
> Part 7— Categorical Data Plots
https://medium.com/@mukul.mschauhan/data-visualisation-using-seaborn-464b7c0e5122 

import seaborn as sns 
acuerdate que pd.describe() te hace un resumen del dataset (es como summary en R)
 
sns.set --> lo primero que hay que poner
sns.stripplot --> para valores categoricos
sns.barplot --> barras
sns.pairplot -- visualización por pares
corr = df.corr()
sns.hearmap(corr, mask=, cmmap=, vmax=)
 
sns.set_style() --> para cambiar el estilo del grafico
                whitegrid cambia el fondo y lo pone blanco       
                ticks pone marquitas en el lateral y cierra en una caja
sns.despine() deja el grafico solo con barra izquierda y abajo
 
sns.axes_style()
 

BOKEH – VISUALIZACION INTERACTIVA
https://towardsdatascience.com/data-visualization-with-bokeh-in-python-part-one-getting-started-a11655a467d4
Puedes ir cambiando con deslizadores las herramientas


## PLOTLY
Crea graficos sofisticados interactivos (util para visualizar datos cruzados)
https://towardsdatascience.com/the-easy-way-to-do-advanced-data-visualisation-for-data-scientists-bbc852e26fb6
https://towardsdatascience.com/its-2019-make-your-data-visualizations-interactive-with-plotly-b361e7d45dc6

## ALTAIR
Visualización interactiva mejor que seaborn y matplotlib

## IPYVOLUME
Para visualizar volumenes en 3D

## PYDOT
Para crear  algoritmos y graficos de nodos etc en py

## BASHPLOTLIB
Para hacer graficos en la consola

## COLORAMA
Colorea la terminal 

## ORANGE
Visualizacion interactiva basada en interfaz grafica (más potente y compleja).
https://towardsdatascience.com/data-science-made-easy-interactive-data-visualization-using-orange-de8d5f6b7f2b