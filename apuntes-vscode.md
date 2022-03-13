# VISUAL STUDIO CODE

## Comandos basicos

### Paleta de comandos
`F1` o `mayus+cmd+P`

`cmd+P` para buscar archivos solo

`?` muestra ayuda

`cmd+K` = comandos de teclado (luego hay que elegir otra tecla)

`cmd+k` y `cmd+S` = keyboard shortcuts

`cmd+k` y `cmd+c/cmd+u` = comment/uncomment (#)


### Varios cursores
`alt+click` con el raton  

`alt+cmd+up/down` en misma linea (teclado)

`mayus+alt` selecciona texto multiple
:   si seleccionamos una palabra con cmd+d vamos seleccionando las siguientes

### Sugerencias (intellispace)

`alt+espacio` detras de un punto (o tab)
cmd y poner encima de algo = información
cmd y pulsar encima de una funcion = va a ella

### Lineas

Mover linea: `alt+up/down`
copiar linea: `alt+mayus+up/down`
eliminar linea: `mayus+cmd+K`
ir a una linea concreta: `ctrl+g`

### Variables

Ponerte encima de la `variable + F2`

### Extensiones interesantes

* Prettier: formats code before saving
* Better comments: change colors for comments to have differents types
* Cascadia code: font very nice for coding
* Polacode: create nice images from code snippets
* Bracket pair colorizer: colorea los () [] para que cuadren por colores
* Import Cost: te dice lo que computacionalmente cuestan los imports
* Markdown PDF: convierte md ta pdf
* Live share: permite crear una sesion de codigo colaborativa con alguien (como si fuera en la nube)
* Github: permite una mayor integracion con github


### Código expansible

Para poder colapsar código: `cmd+K y cmd+0`

### Debug

- En la pestaña debug, le damos al play de la izquierda para empezar debug.
- Ponemos breakpoints con el punto rojo a la izquierda del codigo.
- Dandole a 'step over' nos movemos por el codigo linea a linea
- Cuando los breakpoints saltan, podemos:
  - Visualizar variables (arriba)
  - Visualizar una variable concreta (watch), incluso haciendo operaciones con ella (ej: variable.shape)
    - La podemos añadir con el boton derecho 'Debug Add to watch'


Dentro de los breakpoints, podemos establecer:
- Expression: que solo pare si se cumple una expresion (ej: variable > 7)
- Hit count: que solo pare si pasa un numero de veces por ese punto (ej: para usarlo en bucles)
- Log message: no para la ejecucion pero muestra un mensaje en la debug console

Hay dos tipos de excepciones:
* Raised exceptions: las que nosotros mostramos cuando algo falla en el codigo
* Uncaught exceptions: las que no están contempladas en el codigo


#### Formateo

Para que quede bonito: `cmd+K y luego cmd+F`


#### Errores

Corregir errores: `F8 o mayus+F8`

tambien `mayus+cmd+m`

### Terminal

abrirla = `` ctrl+` ``

clear = `cmd+K` o escribir `cls`

## Markdown

preview: `mayus+cmd+V`

para hacerlo en directo: `cmd+k V`

Extensiones utiles:

* Markdown all in one: para TOC, exportar y sugerencias
* Markdown linting: para limpieza y errores
* Markdown enhanced (no instalada): para comandos avanzados

## python

> Ejecutar código

Para ejecutar código le damos al play

Para ejecutar líneas sueltas las seleccionamos y alt+enter

code cells tipo jupyter 
`# %%` al inicio

> Debugging

Marcamos punto rojo y F5 para debug (se para ahí)

En debug console podemos meter código


> Limpieza

* Format document = `autopep8`  (solo lo pone 'bonito') o `black`
  * Podemos darle a `sort imports`para ordenar los imports.
* linting = mejor aun

> Extensiones interesantes

* AREPL for python = evalua codigo automaticamente mientras escribimos (sin ejecutar) con ctrl+mayus+P
* kite = autocompletado
* autoDocstring = crea un docstring automático
* Python test explorer = crea un explorador de test (para ver que el codigo se ejecuta bien --> con `assert`)
* Qt for python = para diseñar GUIs

## git
`cmd+k cmd+D` ver diff

### Sincronizar preferencias

Darle a settings sync: on, y sincroniza con tu cuenta github toda tu configuracion.
Además podemos ver las máquinas que están sincronizadas.

 