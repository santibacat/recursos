# PYTHON BÁSICO

## Operaciones

Basic operations: +-*/

```python
20 // 6 # floor division
1.25 % 0.5 # modulus
20 ** 3 # exponentiation
x += 1 # same as x=x+1
```

* Can't use `x++` as in C++. Use `x+=1`.
* Don't use `^` for exponentiation as in C++. Use ``**`.
* You can't sum string + ints (`error`). Must use type conversion.


```python
print(2+"2") # error
print(2+int("2")) # 4
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-6-db53bd892eed> in <module>()
    ----> 1 print(2+"2")
    

    TypeError: unsupported operand type(s) for +: 'int' and 'str'


* CAREFUL: Convert float to int **DOES NOT ROUND** (just cuts). Eg: f6.7 > i6


**COMPLEX NUMBERS**

```python
complex_var = 2+3j
# imaginary or real components
print("Complex: ", complex_var, "Real: ", complex_var.real, "Imag: ", complex_var.imag)
```

## Strings

* Create them with `""` or `''`
* Behave like list of characters (can use `str[7]`)
* Use `\` for special chars (\n = new line)


```python
print("Brian\'s mother")
# Print special characters: use \ before
print("Brian\'s\n house")
```

    Brian's mother
    Brian's
     house


Strings are really _char lists_:

* `len` = length
* `string.replace(word, another)` = replace one word for another
* `string[start:end:step]` = char indexing


```python
s = 'Hello world'
print(len(s))
print(s.replace('world', 'test'))
print(s[1:7:1])
```

    11
    Hello test
    ello w


### STRING METHODS

ASDFAJ

### USER INPUT


```python
input("Write something: ")
```

    Write something:  HELLO





    'HELLO'



### STRING FORMATTING


```python
print('str1','str2') #  with space
print('str1'+'str2') # without space
print('price = %.2f' % 3.14) # string format
print('price = {0}'.format(3.14)) # string format2
```

    str1 str2
    str1str2
    price = 3.14
    price = 3.14



```python
# Old way using string.format:
nums = [4,5,6]
print("Numbers: {0}, {1}, {2}".format(nums[0], nums[1], nums[2]))
print("Numbers: %i, %i, %i" % (nums[0], nums[1], nums[2]))

# New way using f before string (string format 3)
print(f"Numbers: {nums[0]}, {nums[1]}, {nums[2]}")
```

    Numbers: 4, 5, 6
    Numbers: 4, 5, 6
    Numbers: 4, 5, 6


# VARIABLES

* Python variables are muteable and don't need to be declared (don't have defined type, can dinamically change <-> int, float, string...
* Delete a variable `del. Eg: `del foo`
* Case-sensitive (hi != Hi). Don't use UpperFirst (use upperLater). Separate with "_". No special chars (ñ, ç, €).


```python
name = "santi"
money = 13.753453
print("Hello, %s! You've got %3.1f new dollars" % (name, money))
```

    Hello, santi! You've got 13.8 new dollars


## STORAGE

* List []
* Dictionary {}
* Tuple ()
    * namedTuple


* Set ()
* 


### LISTS [ ]


```python
l = [] # crea una lista
l.append("a") # añade un elemento
l[1] = 'p' # modifica el segundo elemento
l.insert(0,'i') # inserta un elemento en un indice determinado (aqui, el primero)
l.remove("a") # elimina un elemento por su CONTENIDO
del l[7] # elimina un elemento por su INDICE
sorted(l) # ordena la lista alfabeticamente
l.extend(l2) # añade los valores de la lista l2 a la lista l
len(l) # longitud de la lista
```


```python
l[-2] coge el PENULTIMO valor
len(hola.split()) # contar palabras de una fase
```


```python
#Para UNIR una lista
print "".join(unir) # une con espacio
print ".".join(unir) # los une con punto
```


* Can mix different types
* Can be nested within other lists (multidimensional), better use numpy
* Useful functions: max/min(list), list.count/remove/reverse()
* REMEMBER: First item is list[0] (!= list[1])


```python
words = ["spam", "eggs"]

# Check pertenence: word in words
print(not "spam" in words)

# Add new iem at the end
words.append("last")
# Can't iterate on an empty list, ilke in a for loop --> list[i]. 
# You must fill it first (not [], but [0]*100 p.e.)
words.insert(2, "middle") # Adds item in position #2
print(words)


```

    False
    ['spam', 'eggs', 'middle', 'last']



```python
# For range creation, need to convert to list. Otherwise returns range object.
# range(first, last, interval), REMEMBER STARTS ON 0
print(list(range(10)))
print(range(10))
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    range(0, 10)


__List Slices__

* list[start:stop:step]
* list[::-1] reverses a list

__List comprehensions__




```python
evens = [i**2 for i in range(10) if i**2 % 2 = 0]
print(evens)
```

### TUPLES ( )

son como listas pero inmutables

* Immutable lists. Created with or without ()
* Useful for switching: a, b = b, a



```python
point = (10, 20) # con parentesis
point[0] = 40 # dará error
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-2-83e607a374fc> in <module>
          1 point = (10, 20) # con parentesis
    ----> 2 point[0] = 40 # dará error
    

    TypeError: 'tuple' object does not support item assignment


### DICTIONARY { }

Diccionarios: son listas que guardan los elementos como pares indice-valor (key-value)


```python
params = {'nombre': 'Will', 'apellido':'Smith'} # se crea con corchetes
params['direccion'] = 'Calle Mayor' # añade un nuevo elemento
```

* Used as key:value mappings
* Can use dict.keys(), dict.values(), dict.
* As in lists, can use __in__ and __not in__.
* __dict.get()__ to see if key is in dict; otherwise returns specified value.



```python
ages = {"Dave": 24, "John": 13}
```

## ITERATION

### if-else-elif

Require indentation (!= C++)

```python
statement1 = False
statement2 = False

if statement1:
    print("statement1 is True")
    
elif statement2: #aamw as else if
    print("statement2 is True")
    
else:
    print("statement1 and statement2 are False")
```


```python
n = 3
if n <2:
  print("small") # indentation
elif n < 4:
  print("medium")
else:
  print("big")
```

    medium



```python
# one-line short-if
print("small") if n<4 else print("big")
```

    small


### While

* To end a while loop prematurely, use a __break__ statement.
* To go to the beginning of the loop, __continue__ (in a next iteration)
* To stop manually, ctrl-c
* Infinite loop: `while True:`


```python
i = 0
while True:
   i = i +1
   if i == 2:
      print("Skipping 2")
      continue
   if i == 5:
      print("Breaking")
      break
   print(i)
```

    1
    Skipping 2
    3
    4
    Breaking


### For loops: 

recuerda que al iterar, el rango no contiene el ultimo valor

range(from, to, by) con el to NO incluido


```python
range(4) # dara 0, 1, 2, 3
range(-3,3) # dara -3, -2 ... 1, 2

l1 = [x**2 for x in range(0,5)]
# forma compacta de un for loop, igual a:
for x in range(0,5):
    x**2
```

Para iterar en key-value de un diccionario (dos formas)


```python
for key, value in params.items():
    print(key + " = " + str(value))

for idx, x in enumerate(range(-3,3)):
    print(idx, x) 
    # con ello usamos el indice de iteración (idx = 0, 1, 2...)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-24c71bc069e7> in <module>
    ----> 1 for key, value in params.items():
          2     print(key + " = " + str(value))
          3 
          4 for idx, x in enumerate(range(-3,3)):
          5     print(idx, x)


    NameError: name 'params' is not defined


Enumerate es la forma más sencilla de iterar en listas, con dos valores: el índice de iteración y el valor en sí:


```python
monsters = ['Kraken', 'Leviathan', 'Uroborus', 'Hydra']

for i, monster in enumerate(monsters):
  print(i, monster)
```

    0 Kraken
    1 Leviathan
    2 Uroborus
    3 Hydra


## CONDITIONALS

**BOOLEAN OPERATORS**

* and (only true if all true) `&&`
* or (true if any true) `||`
* not (true if false) `!=`
* operator precedence
  * ** > complement > */%// > +- > boolean
  
![](https://api.sololearn.com/DownloadFile?id=3515)





```python
Si comparamos dos strings con > o < se comparan por diccionario


s1 = "Jennifer"
s2 = "Python"

print (s1 > s2) # True -> since 'Jennifer' comes lexographically before 'Python'

# Checking if list is empty
l1 = []
l2 = ["Jennifer"]

if l1:
    print (1)
elif l2:
    print (2)
```


      File "<ipython-input-45-29548059d031>", line 1
        Si comparamos dos strings con > o < se comparan por diccionario
                    ^
    SyntaxError: invalid syntax



## Funciones

: aconsejable definir 'docstring' (descripción)


```python
def square(x, debug=False):
    """
    Devuelve el cuadrado de un numero
    """
    # aqui debug sería una variable con un valor por defecto
    return x**2
    pass # no hace nada
```

### Lambda functions:


```python
f1 = lambda x: x**2

# es igual que

def f2(x):
    return x**2
```

Podemos 'mapear' una funcion como argumento de otra:


```python
map(lambda x: x**2, range(-3,4))
# Devuelve [9, 4, 1, 0, 1, 4, 9]
```

## EXCEPTIONS

Crear errores:


```python
raise Exception("Descripción del error")
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-4-de58dac3e45d> in <module>
    ----> 1 raise Exception("Descripción del error")
    

    Exception: Descripción del error


Tambien puede, al crearse el codigo, crearlo con dos vias (buena y mala):


```python
try:
    print('Va bien')
except:
    print('No va bien')
    # lo de except se ejecuta si hay algun error
```

* Use __try__ blocks to contain the code
* Use __except__ block for the code if exception occurs
+ Use __finally__ for code that will run anyway at the end of the try block
* Exceptions: ZeroDivisionError, ValueError,
* You can raise your own exceptions with __raise__ Eg.: raise valueError
* __assert__ checks if a statement is True; otherwise gives AssertionError.
* Is useful for checking functions do what you want to do


```python
try:
  print("Hello")
  print(1/0)
except ZeroDivisionError:
  print("Divided by zero")
finally:
  print("Final code")
  assert 2+2==4
  raise ValueError
```

    Hello
    Divided by zero
    Final code



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-6fdd4ab86db1> in <module>()
          7   print("Final code")
          8   assert 2+2==4
    ----> 9   raise ValueError
    

    ValueError: 


## Clases (OOP)

hay que crear primero la clase


```python
class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self, x, y):
        """
        Create a new Point at x, y.
        """
        self.x = x
        self.y = y
        
    def translate(self, dx, dy):
        """
        Translate the point by dx and dy in the x and y direction.
        """
        self.x += dx
        self.y += dy
        
    def __str__(self):
        return("Point at [%f, %f]" % (self.x, self.y))
```

Y luego la instancia de clase:


```python
p1 = Point(0, 0) # this will invoke the __init__ method in the Point class
print(p1)         # this will invoke the __str__ method
p1.translate(0.25, 1.5)
```

### FILE PROCESSING


```python
file = open("filename.txt", "w") #w = write mode, r = read , "wb" = binary mode
file.read(20) # reads 20 lines
file.close() # always need to close the file

# Good practise when working with files, to close always at the end to free mem
try:
  f = open("filename.txt")
  print(f.read())
finally:
  f.close()

# Even better (when with block finishes, file is closed
)
with open("filename.txt") as f:
  print(f.read())
```


```python

```

# DATA SCIENCE

## NUMPY

Libreria de cálculo matricial. Organiza información en arrays

ndarray.ndim = rango (dimensiones)

.shape = longitud de las dimensiones

.size = nº total de elementos de la matriz


```python
import numpy as np
np.array([1, 2, 3], dtype='float')
# es lo mismo que np.ndarray
```

Para subseleccionar (slicing)

a[desde:hasta:salto]

# iPython shell

## Basic Commands


* help() or ? shows information about anything
* ?? shows even more information (source code)
* nameobject.TAB shows object contents (autocomplete), also when importing-using variables
* *character is Wildcard matching (matches any string)



```python
help(len)
len?
len??
len.TAB
*Warning?
```

## Magic Functions

* %paste --> useful when pasting code from internet
* %cpaste --> same but multiline
* %run --> run external code
* %timeit --> get execution time
* %time --> same but only for first loop (useful if first loop is harder to compute)
* %%time --> multiline execution time (must be the first code in block)

* prun --> runs code with profiler (see time for each line, to optimize code)
* %lprun --> line-by-line profiler (must install line_profiler package)
* %memit --> measures memory (must install memory_profiler package)
* %mprun --> measures memory line-by-line profiler (only works for external packages .py)

To get all magic functions>
* %timeit? --> describe a magic function
* %magic --> description of available magic functions
* %lsmagic --> list all magic functions


```python
%cpaste
def donothing(x):
... return x
```


```python
%run myscript.py
```


```python
%timeit L = [n ** 2 for n in range(1000)]
%time L = [n ** 2 for n in range(1000)]
%%timeit
L = []
for n in range(100):
  L.append(n**2)
```

To use history
* In --> last inputs
* Out --> last outputs
* print(_) --> previous output (also usable: __ = _2 = Out[2])
* %history (use: %history -n 1-n)


```python
%history
print(_)
```


```python
%prun print(range(10))

pip install line_profiler
%lprun -f print(range(10))

pip install memory_profiler

```

    range(0, 10)
     

## Terminal commands

* pwd gets current dir
* ls lists current dir
* cat shows inside a document (p.e. untitled.txt)
* ! for use terminal commands in iPython
* can't use !cd. Must use %cd or cd (%automagic function, such as cat, env, ls, man...)

If you save terminal output > python special list (SList) where you can use add functs (grep, fields, s, n, p)

For error debugging:

* %xmode (traces errors when executing code) Eg: %xmode Plain, Context, Verbose
* %debug
* %pdb on (turns on automatic debugger when an exception is raised)


```python
# If you want to add to developing ide >> first line
#!/usr/bin/env python
# change character codification (default ASCII) >> second line
# -*- coding: UTF-8 -*-
```


```python
Terminal commands
* Para listar directorios: `ls`
* Para mostrar el interior de un documento sin abrirlo: `cat`
* Para añadir el entorno de desarrollo a un archivo python (primera linea): `#!/usr/bin/env python`
* Para cambiar la codificacion de caracteres (por defecto ASCII), metemos en la segunda línea del archivo python: `# -*- coding: UTF-8 -*-`
```


      File "<ipython-input-25-da50eef9cd07>", line 1
        Terminal commands
                        ^
    SyntaxError: invalid syntax



## Package organization

### Importing

* Package help: ```help(package)```
* Package functions: ```dir(package)```

* Import package:


```python
import math # imports everything
math.sqrt(2)
from math import * # imports everything as local
sqrt(2)
from math import sqrt as square_root # imports everything as other name local
square_root(2)
```




    1.4142135623730951




```python
Para ver la lista de simbolos de un paquete: `dir`.

    import math
    dir(math)
```


      File "<ipython-input-24-9a7073640286>", line 1
        Para ver la lista de simbolos de un paquete: `dir`.
               ^
    SyntaxError: invalid syntax




```python
Para ver la ayuda de un paquete `help`.

    help(math.sqrt)
```

# BIBLIOGRAFÍA

[Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)


```python

```
