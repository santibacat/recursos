# BASIC PYTHON

## OPERATIONS

Basic operations: `+ - * /`

```python
20 // 6 # floor division
1.25 % 0.5 # modulus
20 ** 3 # exponentiation
x += 1 # same as x=x+1
```

* Can't use `x++` as in C++. Use `x+=1`.
* Don't use `^` for exponentiation as in C++. Use `**`.
* You *can't sum string + ints* (error). Must use type conversion (casting): `float(), int(), str()...`


```python
print(2+"2") # error
print(2+int("2")) # 4
```



* CAREFUL: Convert float to int **DOES NOT ROUND** (just cuts). Eg: f6.7 > i6



**COMPLEX NUMBERS**

```python
complex_var = 2+3j

# imaginary or real components
print("Complex: ", complex_var, "Real: ", complex_var.real, "Imag: ", complex_var.imag)
```

## STRINGS

* Create them with `""` or `''`
* Behave like list of characters (can use `str[7]`)
* Use `\` for special chars:
	* Separators: `\n` = new line, `\t` = new tab
	* `\" \'` = " and ' in strings


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

* `title()` Title mode
* `upper()` UPPERCASE
* `lower()` lowercase
* `strip()` no spaces (`lstrip()` left and `rstrip()` right)
  * `.strip("/")` eliminar caracteres indeseados al final del string
* `string.printable` to see all printable chars
* `"-".join(sentence)` unir listas de string
* `sentence.split()` separar por caracteres
* `len(string.split())` counts chars
* Para encontrar el elemento más frecuente de una lista: `print(max(set(text), key=test.count))`

Search on strings:

* `count(pattern)` count the number of times a string is repeated in a string
* `find(pattern)` return index for FIRST pattern.
* `replace(pattern, newpattern)` replaces one word with another

### USER INPUT


```python
input("Write something: ")
```


### STRING FORMATTING

Print style:

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


Old way for string formatting: `{}.format()`.

```python
nums = [4,5,6]
print("Numbers: %i, %i, %i" % (nums[0], nums[1], nums[2])) # old, don't use
print("Numbers: {0}, {1}, {2}".format(nums[0], nums[1], nums[2]))
	# acceptable, numbers needed only if repeated; otherwise in order
```

New way **f-strings** using `f` before string (after python 3.6):

`print(f"Numbers: {nums[0]}, {nums[1]}, {nums[2]}")`


Es preferible usar double_quotes para que no afecte a las variables que haya dentro:

`f"My name is {person['name']}"`

* Para formatear números se usan _dos puntos_: `{n:02.3f}`

* Para formatear fechas igual: `{birthday:%B %d %Y}`

## VARIABLES

* Python variables are muteable and don't need to be declared (don't have defined type, can dinamically change <-> int, float, string...)
* Delete a variable `del`. Eg: `del foo`
* Case-sensitive (hi != Hi). Don't use UpperFirst (use upperLater). Separate with "_". No special chars (ñ, ç, €).


```python
name = "santi"
money = 13.753453
print("Hello, %s! You've got %3.1f new dollars" % (name, money)) # old C-type formatting
```

    Hello, santi! You've got 13.8 new dollars


More advanced topics: see [variable scope](#variable-scope)




## CONDITIONALS

**BOOLEAN OPERATORS**

In python, better use _words_ rather than symbols. Parenthesis use as in math.

* and (only true if all true) `&&`
* or (true if any true) `||`
* not (true if false) `!=`
* operator precedence
  * `**` > `complement` > `* / % //` > `+ -` > `boolean`
  
![](https://api.sololearn.com/DownloadFile?id=3515)




## ITERATION

### IF-ELSE-ELIF

Requires indentation (!= C++)

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


### WHILE

* Infinite loop: `while True:`
* To end a while loop prematurely, use a `break` statement.
* To go to the beginning of the loop, `continue` (in a next iteration)
* To stop manually, `ctrl-c`



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


### FOR

Iterates in a `range(from, to, step)`. From = 0 to = n-1


```python
# long
for x in range(0,5):
    x**2
# compact 
l1 = [x**2 for x in range(0,5)]
```

With `enumerate` we can get the index value of the list iteration (use `start` to set the start id).

```python
for index, value in enumerate(list, start=1):
  print(index, value) # prints id from 1
```



## DATA STRUCTURES

Types: ['List []'](#lists), ['Tuples ()'](#tuples) (and [namedTuples](#namedtuples)), ['Sets {}'](#sets)
 and ['Dictionary {}'](#dictionary)

Usage:

* Lists: when modified frequently and don't need random access.
* Dicts: when need key-value pair, need fast lookup, data constantly mofified
* Sets: when need uniqueness
* Tuples: when don't want change in data


### LISTS

* Unordered, can mix different datatypes
* Can be nested within other lists (multidimensional), but better use numpy
* REMEMBER: First item is list[0] (!= list[1]) --> _Zero-index_
* Be careful when `=` two list values, because they are assigned to the SAME OBJECT.

New list: `l = []`.	

**List Indexing:**

* `list[start:stop:step]`
* `list[::-1]` reverses a list :star:
* `list[-2]` takes the penultimate value
* `list[:]` takes all list WITH COPY

```python
# Behaviour of copying or not copying
# Copying a list
l1 = ["We", "should", "use", "[:]", "to", "copy", "the", "whole", "list"]
l2 = l1[:]
l2.append(". Using [:] ensures the two lists are different")
print(l2)
print(l1)
# RESULT = both are diferent
l2 = l1
l2.append(". Using [:] ensures the two lists are different")
print(l2)
print(l1)
# RESULT = both point to the same list
```


**Useful functions:**

* `len(list)` get number of list elements
* `list.append(item)` appends at the end :star:
* `list.insert(index, item)` appends at given position
* `list.extend(list2)` appends **list2** contents to existing list (don't use append)
* `list.remove(item)` removes by content ONLY THE FIRST OCURRENCE.
* `del list[8]` deletes element by index :star:
* `list.index('value')` returns the index for one value in the list
* `sorted(list)` sorts alphabetically TEMPORARY. `list.sort()` sorts PERMANENTLY.
  * May use `sort(reverse=True)`
* `list.reverse()` reverses PERMANENTLY.
* `list.pop()` removes last element and returns it
* `union.join(list)` joins all elements with <union>. Eg. `"-".join(list)`.
* `sum/max/min(list)` si lista numérica solo.
* `list.count(value)` cuenta las veces que se repite un valor en una lista :star:

Exception: `list.insert(-1, item)` inserts in the PENULTIMATE position, not the last. Use `append` instead.
    
    # INSERT
    my_list = ['A', 'B', 'C', 'D']
    my_list.insert (-1, 'E')
    print (my_list)
    # result = ['A', 'B', 'C', 'E', 'D']

    # POP: pop(n) -> Removes the element at index 'n' and returns it
    l1 = ['A', 'B', 'C', 'D', 'E']

    # Removes the element at 0 position and returns it
    c = l1.pop(0)
    print (l1)
    print (c)
    # Works as expected with negetive indexes
    c = l1.pop(-1)
    print (l1)
    print (c)


**Sort objects**:

Podemos usar `sorted()` que crea nueva variable o el método `list.sort()` para una ya creada.

* `reverse=True` la ordena al revés
* `key=` es una función que diga *cómo* ordenar. 
	* Puede ser un lambda tipo `key = lambda e: e.salary` o un getter `key = attrgetter('age')`.

```python
    l1 = ['E', 'D', 'C', 'B', 'A']
    print(sorted(l1))
    print(l1)
    l1.sort()
    print (l1)
```

**Check pertenence**:

```python
words = ["spam", "eggs"]
print(not "spam" in words)
```


For **range creation**, need to convert to list. Otherwise returns range object.
`range(first, last, interval)`. Remember: starts from 0 to n-1

```python
# range(first, last, interval), REMEMBER STARTS ON 0
print(list(range(10)))
print(range(10))
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    range(0, 10)





### TUPLES

* Immutable lists. Created with or without `( )`
* Useful for switching: a, b = b, a


```python
point = (10, 20) # same as 10,20 without parenthesis
point[0] = 40 # give error, ca't change
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-2-83e607a374fc> in <module>
          1 point = (10, 20) # con parentesis
    ----> 2 point[0] = 40 # dará error
    

    TypeError: 'tuple' object does not support item assignment


### namedTuples

Enhanced and more comprehensive tuples. Same advantages (non-mutable and faster than dict).

Eg. RGB-colored tuples (to identify each color)


```python
from collections import namedtuple
# namedtuple('name', ['args'])

Color = namedtuple('Color', ['red','green', 'blue'])
new_color = Color(55,155,255)
new_color.blue # we can access individually
```

### SETS

* Similar to lists but ORDERED and UNIQUE values (no duplicates). `{ }`
* Can't create with {} (it's a dict)

VERY USEFUL: we can use several sets and play with IoU:

* Intersection `&`: combines both
* Union `|`: only items in both
* Difference `-`: only items in first but not in second
* Simmetric difference `^`: items in either set but not in both




### DICTIONARY

* Used as key:value mappings `{ }`
* Can use `dict`, `dict.keys()`, `dict.values()`
* As in lists, can check pertenence with `in` and `not in`.
* `dict.get()` to see if key is in dict; otherwise returns specified value or None.
* `sorted(dict.keys())` useful for iterating over sorted dictionary keys

Create with `dict = {}`. Access one key with `dict["keyname"]`.

```python
params = {'name': 'Will', 'surname':'Smith'} # creation
params['address'] = 'Major st.' # adds one element
```

**Iteration over dictionaries:**

1) Iterating over `key-value` pair:

```python
for key, value in dict.items():
    print(key + " = " + str(value))

# same as
for key in dict.keys():
    print(key = dict[key])
```

2) Iterating with temporary ID with `enumerate` (uses iteration index and value):

Usage:

```python
for idx, x in enumerate(dict):
    print(idx, x) 
    # we get iteration index (0,1,2,3,...) and the value of iteration.
```

Eg:

```python
monsters = ['Kraken', 'Leviathan', 'Uroborus', 'Hydra']

for i, monster in enumerate(monsters):
  print(i, monster)
```

    0 Kraken
    1 Leviathan
    2 Uroborus
    3 Hydra

#### defaultDict

crea un valor por defecto para cuando inicializas un diccionario sin key por primera vez.

### COMPREHENSIONS

Generalmente se usan listas: **list comprehensions**

```python
[n^2 for n in nums if n%2 == 0]
[(letter,num) for letter in 'abcd' for num in range(4)]
```

Tambien se pueden hacer con diccionarios y sets:

```python
{name:hero for name, hero in zip(names, heros)if name != 'Peter}
```

Tambien con sets que tienen valores únicos:

```python
{n for n in nums}
```

Los generadores son muy similares a las comprehensions.

Recuerda que `zip` crea tuplas con el primer indice de cada lista, luego el segundo, y así hasta el final.







## FUNCTIONS

Recommended to write 'docstring' with `"""` at the beginning. To call functions you need () at the end `function()`.


```python
def square(x, debug=False):
    """
    Returns de square of a number
    """
    # debug has a default value
    return x**2
    pass # don't do anything
```

Blank function = use `pass`.


`*` in args means to make a tuple with arguments:

* Has to be placed last
* Can't have more than one (eg `def ...(..., *arg1, *arg2)` --> gives error)
* Can have many keyword arguments after = `(..., *args, **kwargs)`
* Kwargs return a *dictionary* with key_value pairs.

BE CAREFUL: you **can't** pack args/kwargs because will handle them as args. You need to SPECIFY 

```python
courses = ['math', 'science']
info = {'name': 'John'; 'age': 28}
student_info(courses, info) # INCORRENT, will both as ARGS
student_info(*courses,**info) # THIS IS CORRECT
```

**Recursion** means calling a function inside itself. Eg: `factorial`

* Need to have a base_case, when you can't break sub-functions any further (exit-condition of the recursion).


### Lambda functions:

Short functions

```python
f1 = lambda x: x**2

# same as

def f2(x):
    return x**2
```

### Map/Filter/Zip

We can use **map** to apply a function to many iterables (apply the same function to each object, eg: list):

`map(function, object)`


```python
map(lambda x: x**2, range(-3,4))
# Devuelve [9, 4, 1, 0, 1, 4, 9]
```

**Filter** removes items that don't match a boolean predicate:

`filter(predicate, object)`

```python
filter(lambda x: x%2==0, nums)
```

To print result convert to `list()` first.

**zip** packs two lists to iterate simultaneously:

```python
for x, y in zip(first, second):
  print(x+y)
```

### Counter

Sirve para contar cosas (por ejemplo listas, y te da una tupla con cada valor y su conteo.

```python
from collections import Counter
ages = [22, 22, 25, 25, 30, 24, 26, 24, 35, 45, 52, 22, 22, 22, 25, 16, 11, 15, 40, 30]
value_counts = Counter(ages)
print(value_counts.most_common())
```


## ERRORS AND EXCEPTIONS

* Use `try` blocks to contain the code to test for exception
* Use `except` block for the code if some exception occurs
* Use `else` block for code if no exception occurs in try block
+ Use `finally` for code that will run anyway at the end of the try block
* Exceptions: `ZeroDivisionError`, `ValueError`, `TypeError`, `FileNotFoundError`...
* Use `raise(ExceptionType)` to raise an exception manually
* `assert` checks if a statement is True; otherwise gives AssertionError.
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

Raise exception:

```python
raise Exception("Error Description")
```

Create custom ErrorType:

```python
class myError(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)
```


## FILE PROCESSING

First `open(file)` and then `file.read()` / `file.write()`.

```python
file = open("filename.txt", "w") #w = write mode, r = read , "wb" = binary mode
file.read(20) # reads 20 lines
file.close() # always need to close the file
```
Good practise when working with files, to close always at the end to free mem.
Even better, read in the `else` block.

```python
try:
  f = open("filename.txt")
except:
    pass
else:
  print(f.read())
finally:
  f.close()
```

Better way: `with` keeps file open as long as needed (closes automatically):

```python
with open("filename.txt") as f:
  print(f.read())
```

# CLASSES - OOP

> Utilidad: nos permite reutilizar más eficientemente el código.

* _CLASES_ = defición de algo
	* _OBJETOS_ (= instancias de clase)
	* Poseen *atributos* (= variables) y *métodos* (= funciones)

```{python}
class Employee:
	pass

```
**RESUMEN**


* Tipos de variables:
	* Instance variables
	* Class variables
* Tipos de métodos:
	* Métodos de instancia
	* Métodos estáticos
	* Metodos de clase


A la hora de acceder:

* no usaremos paréntesis si es un atributo: `employee1.fullname`
* usaremos paréntesis si es un método: `employee1.fullname()`


## VARIABLES (ATRIBUTES)

Tipos de variables:

* Instance variables
	* Contienen datos únicos de cada instancia
* Class variables
	* Variables conjuntas para todos los objetos (con el mismo valor)
	* Se ponen al principio de la clase
	* Se puede acceder desde la clase y desde las instancias

`object.__dict__` = te da un diccionario con todos los atributos de un objeto.

### Instance variables

Contienen datos únicos de cada instancia.  
Ej: `employee1.name, employee1.email`

Se pueden modificar.


### Class variables (variables de clase)

Son variables que pertenecen a la clase y no a los objetos (son conjuntas y tienen el mismo valor para todos los objetos).  
Se ponen antes del init de la clase (al principio).
Ej: `pi` en la clase circulo. Se podrá acceder con `Circulo.pi` y con `circulo_uno.pi`.

No necesitan instanciar el objeto antes.

En python las variables de clase **NO SON INMUTABLES**; por ello se le suele poner un guion bajo delante. `_pi` (por convencion).

#### (Avanzado) Acceso a variables de clase
No se pueden acceder directamente desde un método, sino que habría que acceder a través de la clase o a través del objeto (`Employee.company o self.company`, nunca solo con `company`).

La forma de acceder es importante porque:

* Si cambiamos el valor para la clase, afecta a todos los objetos. `Employee.raise_amount = 1.05`
* En cambio si cambiamos el valor para un objeto no afecta a toda la clase. `employee1.raise_amount != Employee.raise_amount`.

Esto es importante al programar un método o una subclase, ya que si accedemos con `self.raise_amount` nos referiremos a un valor que se asigna automáticamente sólo al objeto (sobreescribe al de la clase para ese objeto) y si accedemos con `Employee.raise_amount`nos referimos a la clase.  
Es interesante usar una variable de clase para valores inmutables por los objetos como `Employee.num_of_employes`).




### Private atributes (Atributos PRIVADOS)

Weakly-private atributes: 

* Se ponen con una `_`. Ej: `self._hiddenlist`
* Simplemente advierten de que no deberían usarse por código externo a la clase (pero se puede acceder).
* Su único efecto es no ser importados si usamos `from module_name import *`.

Strongly-private atributes:

Los atributos privados solo pueden ser modificados por la clase y no por fuera (ni siquiera por la instancia de clase).
Esto se hace haciendo __ delante del atributo al crearlo.
Ej `self.__password = password`

Para poder acceder desce fuera de la clase debemos usar `object.Class.__privatemethod`.

Eg: `audrey.Dog.__isanimal`



####  Limitar atributos de una clase

`object.__dict__` = te da un diccionario con todos los atributos de un objeto.

En una clase o en una instancia de clase podemos añadir los atributos que queramos (aunque no estén definidos).
Pero a veces estos atributos provocan errores; para ello podemos definir que una clase tenga permitido usar unos atributos definidos.

Para ello pondremos esto al principio de la clase:
`__slots__ = ['atributo1', 'atributo2']`

Esto además provoca que no podamos usar `__dict__`.
El atributo `__slots__` no se hereda (las subclases podrán establecer atributos que quieran).


### Property decorators (decoradores)

Los property decorators permiten definir metodos y acceder como si fueran atributos.
`@property`.

Al definir como property ya no podemos cambiar el valor de los atributos desde fuera como si fueran normales, es necesario modificarlos con otros decoradores (setters y getters).


Por ello principalmente sirven para trabajar con los atributos privados de una clase (ej self.__password) para no manipularla directamente:

```python
@property
def password(self):
	return self.__password
	
@password.setter
def password(self, valor):
	self.__password = self.__generar_password(valor)

@password.getter
def get_password(self, valor):
	print(self.__password)
	
@password.deleter
def delete_password(self):
	self.__password = None
	print("Password deleted")
```


### VARIABLE SCOPE

Es el alcance de cada variable en python

*LEGB* es en orden en que python busca:

* Local
* Enclosing
* Global
* Built-in

_Enclosing-scope_ se refiere a que todo lo interior incluye a lo exterior, pero no al revés.


Podemos especificar que dentro de una funcion se use un valor global y no un valor local de dentro de la funcion:

```python
x = 'outside x'

def test():
	global x
	x = 'inside x'

print(x) # = 'inside x'
```

Lo mismo podemos hacer en funciones anidadas, cambiar un valor más externo pero sin afectar al valor local

```python
def outer():
	x = 'outer x'
	def inner():
		nonlocal x
		x = 'inner x'
	inner()
outer() # dará 'inner x'
```

#### Built-in variables

_Built-in_ se refiere a nombres globales de python.
Tenemos que tener cuidado porque podemos sobreescribirlas con nuestras propias funciones y no da error.

```python
import builtins
print(dir(builtins))
```


---------







## METHODS


* Métodos de instancia (regular methods)
	* contienen self
	* pueden motificar el objeto
	* pueden acceder y modificar a la clase
* Metodos de clase (class methods)
	* cls
	* no pueden modificar el objeto
* Métodos estáticos (static methods)
	* no contienen self ni cls

	




### __ init __ method (constructor)
Se ejecuta automáticamente al instanciar una clase.
Args: self (por convención) +  demás + *args + **kwargs.


```
def __init__(self, first, last):
	self.first = first
	self.last = last
```

### Regular methods (métodos de instancia)
Cada metodo dentro de una clase tiene que tomar la instancia como primer argumento (self).

```
def fullname(self):
	return f'The name is {self.name} {self.surname}'
```
Explicación: es como si llamaramos a la clase con el argumento como objeto
`Employee.fullname(employee1) == employee1.fullname()`



### Class methods (métodos de clase)
> @classmethod

Utilizan la clase como argumento, no el objeto; por ello hay que usar `cls` (no self).

Pertenecen a la clase, y pueden usar los atributos y métodos públicos de las clases padre (a diferencia de los métodos de clase, que solo pertenecen a la clase y nada más).


No se suelen llamar metodos de clase desde un objeto, pero hace lo mismo que llamarlo desde una clase.
`Employee.set_raise_amount(1.05) == employee1.set_raise_amount(1.05)`
Ambas nos lo cambian para todos (ya que cambian la variable de clase).


#### Constructores con classmethods
Mucha gente usa métodos de clase para crear constructores con funciones. Ej.

```
@classmethod
def from_string(cls, string):
	name, surname = string.split('-')
	return cls(name, surname)
```

La `cls` del final se refiere a la clase, y es lo mismo que si pusieramos `Employee`. De esta forma estamos creando la clase con un método:  
`employee1 = Employee.from_string('Santi-Bacat')`


### Static methods (métodos estáticos)

> @staticmethod


No pasan ningun argumento al método (ni self ni cls) a diferencia de los anteriores. Son como funciones normales pero con alguna conexión con la clase.

No se pueden atribuir a un objeto real (ni a la clase), pero son útiles para la clase.
Le pertenecen a la clase, no a la instancia.

Ej: un método que calcula si es festivo o no para los empleados (no usa nada de la clase).

---


## MAGIC METHODS
Son `__methods__` y sirven para establecer acciones por defecto.

**Representación del objeto:**

`__repr__`: unambiguous representation  (solo para que la vean otros desarrollaodres).
`__str__`: readable representation  (para el usuario final).
> Por defecto se usa str>repr cuando hacemos un print

**Operaciones:**
`__add +, sub -, mul *, matmul, truediv /, floordiv //, mod %, pow **__` = para hacer cálculos con clases
Para el cálculo se hace usando other para el otro argumento:

```
def __add__(self, other):
	return (self.pay + other.pay)

# same as x.__add__(y)
```

Tambien se pueden hacer con booleanos: `__and &, xor ^, or |__`.

y comparaciones: `__lt <, le <=, eq ==, ne !=, gt >, ge >=__`

y conversión de objetos: `__int__`= para convertir a entero

NOTAS:

* Si x-y son de distintos dipos y `__add__` no está implementada, se usa la opuesta (`y.__radd__(x)`)
* Si `__ne__` no está implementada, devuelve el contrario de `__eq__` 



Otros sirven para hacer que las clases sean como contenedores:

* `__len__` = calcular la longitud
* `__getitem__` para obtener el valor al indexar
* `__setitem__` para cambiar valores indexados
* `__delitem__` para dliminar valores indexados
* `__iter__` para iterar en objetos
* `__contains__` para hacer `in` sobre el objeto
* `__new__` = se ejecuta antes que __init__ (lo primero de todo)
* `__str__` y `__repr__` = para representar el objeto (lo que vemos al hacer print).  [Ver más](#str-vs-repr).
* `__getattr__` = se usa cuando un atributo no está definido, y en vez de dar error, hace algo (crea el atributo, muestra un mensaje, etc). [Más info](https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute).



------



## INHERITANCE (HERENCIA)

Nos permite obtener las características de la clase padre, superponerlas o añadir nuevas.

Primero busca el init en la subclase y luego va a la clase padre.

* Si no lo encuentra, ejecuta el init de la clase padre.
* Si lo encuentra, lo ejecuta aquí. Si hay argumentos repetidos, lo pasamos a la clase padre (para mantener código reproducible). `super().__init__(arg1repetido, arg2repetido)`
  * Tambien podríamos poner el nombre de la clase, pero esto se usa más en herencias múltiples:  `Employee.__init__(arg1repetido, arg2repetido)`



Podemos usar la funcion help para ver todo sobre la clase (la subherencia, atributos heredados...).
Podemos usar `isinstance()` para ver un objeto es instancia de una clase: `isinstance(manager1, Developer)`
Podemos usar `issubclass()` para ver si una clase es subclase de otra (<--): `issubclass(Manager, Employee)` es True.


NO se puede usar herencia circular en python


### Herencia multiple:

```
class Subclase(Clase1, Clase 2):
	return
```

Si un método es privado (__cazar) no puede ser accedido por las claes hijas
El orden de la herencia múltiple es de izquierda a dercha (para override de métodos)




---



# iPYTHON SHELL

## Basic Commands


* `help()` or `?` = info
* `??` more info (source code)
* `nameobject.<TAB>` autocomplete (object content, variables..)
* `*character` = Wildcard matching (matches any string)

```python
help(len)
len?
len??
len.TAB
*Warning?
```

## Magic Functions

To get all magic functions>
* `%timeit?` --> describe a magic function
* `%magic` --> description of available magic functions
* `%lsmagic` --> list all magic functions

Available magic functions:
* `%paste` --> useful when pasting code from internet
* `%cpaste` --> same but multiline
* `%run` --> run external code
* `%timeit` --> get execution time
* `%time` --> same but only for first loop (useful if first loop is harder to compute)
* `%%time` --> multiline execution time (must be the first code in block)

Others:
* `prun` --> runs code with profiler (see time for each line, to optimize code)
* `%lprun` --> line-by-line profiler (must install line_profiler package)
* `%memit` --> measures memory (must install memory_profiler package)
* `%mprun` --> measures memory line-by-line profiler (only works for external packages .py)


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

To use **history**
* `In` --> last inputs
* `Out` --> last outputs
* `print(_)` --> previous output (also usable: __ = _2 = Out[2])
* `%history` (use: %history -n 1-n)


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

## Terminal

* `pwd` gets current dir
* `ls` lists current dir
* `cat` shows inside a document (p.e. untitled.txt)
* `!` for use terminal commands in iPython
    * can't use !cd. Must use %cd or cd (%automagic function, such as cat, env, ls, man...)

If you save terminal output > python special list (SList) where you can use add functs (grep, fields, s, n, p)

* Para añadir el entorno de desarrollo a un archivo python (primera linea): `#!/usr/bin/env python`
* Para cambiar la codificacion de caracteres (por defecto ASCII), metemos en la segunda línea del archivo python: `# -*- coding: UTF-8 -*-`


For **error debugging**:

* `%xmode` (traces errors when executing code) Eg: %xmode Plain, Context, Verbose
* `%debug`
* `%pdb on` (turns on automatic debugger when an exception is raised)


---

# ADVANCED PYTHON

## PACKAGES

### IMPORTING

[Documentation](https://realpython.com/python-modules-packages/) and [Useful tips when importing](https://realpython.com/absolute-vs-relative-python-imports/)


* Package help: `help(package)`
* Package functions: `dir(package)`
* Import package:

```python
import math # imports everything
math.sqrt(2)
from math import * # imports everything as local
sqrt(2)
from math import sqrt as square_root # imports everything as other name local
square_root(2)
```

Al importar con `import *` se importan todos los objetos excepto los que empiezan con `__` como `__name__`. Pero nn los paquetes no ocurre así, y habría que crear además en `__init__.py` una lista para `__all__` con los nombres de los modulos a importar.

#### IMPORTAR DE OTRO DIRECTORIO

Si tenemos un paquete que queremos importar en otra localización:

```python
# Add directory to path
import sys; sys.path.insert(0, '/path')
```

### if name == main

```python
if __name__ == `__main__`:
	main()
```

Sirve para verificar si un archivo está siendo ejecutado directamente (llamando a ese archivo) o es importado.

* Si se llama directamente, lo que hay dentro de `main()`se ejecuta gracias a esta línea. Además tambien se ejecuta todo lo que esté fuera.
* Si se importa ese archivo a otro (`import primerpaquete`) se ejecutaría todo lo que está fuera de `main()`. Por ello, si hay algo que no queremos que se ejecute al importarlo, lo ponemos dentro de `main()`.
	* Ej: si solo queremos importar paquetes o algo para reutilizar código.
* Si queremos ejecutar este código en el otro archivo lo podremos llamar usando `primerpaquete.main()`.


### PACKAGING


To make packages, place all files in the same directory and create a file `__init__.py`, `setup.py` and `README`-`LICENSE`.txt

In `__init__.py` hay que poner los subpaquetes para que se importen automaticamente al importar:
```python
# Ejemplo de __init__.py de un paquete llamado utils
from utils import utils_basic, utils_advanced
# para importar todos, lo mejor es hacer:
from utils import *
```


In `setup.py` you need necesary info for package assembly with `pip`:

```python
from distutils.core import setup

setup(
  name='PackageName',
  version='1.0dev',
  packages=['package1',],
  license='MIT',
  long_description=open('README.txt').read(),
)

After that: OR upload to PyPI, OR create binary with `python setup.py bdist` and after `register` and then `upload`; later `install`.

To create exe: PyInstaller (win-osx) or py2exe (win).

```

---

## BE PYTHONIC

**Duck typing**

Se refiere a no tener en cuenta el tipo de objeto si puede hacer lo que se le permite.

> Si anda como un pato y vuela como un pato, se le trata como a un pato.


**EAFP (*easier to ask forgiveness than permission*)**

Se refiere a que en otros lenguajes vamos anidando condicionales para hacer comprobaciones antes de ejecutar (*LBYL = look before you leak*), y en python es mejor dejar ejecutar y usar try/except para ver los errores.

**PEP8** (Python enhancement proposals) = style guide:

* modules, Classes, variables_with_underscores, CONSTANTS
* Avoid `from package import *`
* Only one statement per line; lines < 80 chars

Enlaces: [PEP-8 python](https://www.python.org/dev/peps/pep-0008/), [Best guilelines medium](https://towardsdatascience.com/best-python-practices-for-data-scientists-11056edda8c7)


---

## DECORATORS

Modify functions using other functions (useful for extending functionallity without modifying a function).

Primero habla de los **closures** que yo no habia oido hablar; creo que son funciones anidadas en la que se guarda la ejecución pero no se ejecuta:

```python
def outer_function(msg):
	def inner_function():
		print(msg)
	return inner_function

hi_func = outer_function('Hi')
```

De esta forma se crea la función con los parametros pero aun no se ejecuta.

Esta es la base de funcionamiento de los decoradores, que modifican la ejecución de una función (wrapped):

```python
def decorator_function(original_function):
	def wrapper_function(*args, **kwargs):
		-loquequeramosquehagaeldecorador-
		return original_function(*args, **kwargs)
	return wrapper_function

def decorator_class(object):
	def __init__(self, original_function):
		self.original_function = original_function

	def __call__(self, *args, **kwargs):
	-loquequeramosquehaga-
	return self.original_function(*args,**kwargs)
```
La forma tipica de los decoradores es `@`. Estas dos cosas son lo mismo:
`@decorator_function de display == decorator_function(display)`

Para preservar la información de las variables, cuando vamos a usar varios decoradores lo mejor es usar:

```python
from functools import wraps

def decorador(orig_func):
	...
	@wraps(orig_func)
	def wrapper(...):
		...
	return wrapper

@decorador
orig_func()
```

### Decorators with args

Se hace añadiendole otra función anidada "prefijo" que irá por encima del decorador:

```python
def prefix_decorator(prefix):
	def decorator_function(original_function):
			def wrapper_function(*args, **kwargs):
					...
		return wrapper_function
	return decorator_function

@prefix_decorator('LOG: )
def function():
	pass
```

Pocas veces se usará pero es interesante saberlo.

---


## GENERATORS

* Iterator function, uses `yield` (not return).
* Generates one object at a time (on-demand: uses less memory).
* Use `next` to get next result.
* When finishes iterator, gives `StopIteration` and you can't get more results.




Cuando hacemos un for con una list comprehension en el fondo estamos haciendo un generador; para ello debemos usar `()` paréntesis.
`(x*x for x in [1,2,3,4,5])`



## ITERTOOLS

Functions useful in functional programming:

* `count(start, [step])` = counts up from num to infinity
* `cycle(object)` = iterates through an iterable infinitely
* `repeat(object, times)` = iterates through an iterable 'x' number

* `accumulate(object)` = sum al previous values in an iterator (p, p+p2, p+p2+p3...)
* `chain(obj1, obj2)` = combines deveral iterables
* `takewhile(predicate, object)` = iterates while predicate is True

* `product(obj1, obj2)` = product of ALL posible combinations.
* `permutations(object)` = permutate ALL posible combinations-order.





---


## RANDOM

`import random`

`random.random()`= valor entre 0 y 1
`random.uniform(inicio, final)` = devuelve float
`random.randint(inicio, final)` = devuelve enteros
`random.choice(list)` = devuelve un valor aleatorio de entre una lista
`random.choices(list, k=veces)` = devuelve una lista de `k` valores aleatorios de entre una lista.
* `weights = [50, 25, 25]` para ponderar el peso de cada elemento original de la lista 

Valores unicos (evitar repeticion), no se usa choices sino `random.sample(list, k=)`.


## REGULAR EXPRESSIONS

`import re`

Generalmente comunes para todos los lenguajes de programación.

Los _raw strings_ se utilizan con `r''` e interpretan en crudo los strings.

Lo primero que tenemos que hacer es crear un **patrón**:
`pattern = re.compile(r'')`

`finditer` busca en el texto y devuelve indices si es igual.

* `pattern.finditer(text_to_search)`


Los **metacarácteres** deben llevar \ delante para diferenciarlos de los caracteres reales. Son:
`. ^$ * + ? {} [] \ | ()`

Las minusculas son un valor y las mayusculas el contrario
| Digit | Value |
|-------|-------|
\d | Digit 0-9
\D | Not Digit (0-9)
\w | Word chars (a-z, A-Z, 0-9, _)
\W | Not word chars
\s | Whitespace (space, tab, newline)

\b | Word Boundary
\B | Not word boundary
^  | Beginning of a String
$  | End of a String

[] | Matches only chars in brackets
[^]| NOT matches inside brackets
.  | Matches anything

Cuantificadores:
*  | 0 o más
+  | 1 o más
?  | 0 o 1
{x}| número exacto
{x,y}| rango


Podemos especificar rangos entre guiones:
`[a-e0-5]` o `[a-zA-Z0-9]` pero solo buscaria UN DIGITO cada vez.

Ej: `r'\d\d\d[-.]\d\d\d[-.]\d\d\d\d'`
Esto buscaría números de teléfono tipo 111-222-3333 o 111.222.3333

Con los cuantificadores podemos evitar repeticiones. Se colocan DETRÁS:
`r'\d{3}.\d{3}.\d{3}'`

Tambien sirven para condiciones (? = tanto si se cumple como si no):
`r'Mr\.?\s[A-Z]\w+'` = buscaría tanto Mr como Mr. seguido de espacio y el nombre en mayuscula (ej: Mr. Bacat o Mr Bacat), pero no si solo fuera la inicial (no buscaría Mr. B)

Los **grupos** nos permiten buscar patrones alternativos pero parecidos. Se usan entre parentesis y con |:
`r'M(r|s|rs)'` buscaría Mr/Ms/Mrs


Cuando encontramos un patrón con
`matches = pattern.finditer(text)` podemos acceder a los distintos grupos que hemos buscado en el texto:

* `matches.group(0)` devuelve todo el string
* `matches.group(1)` devuelve el primer grupo (y así sucesivamente)

Además de `finditer`tambien podemos usar:

`sub` que sustituye el patrón encontrado por un string que le pasemos: `pattern.sub(sub_pattern, text)`
  Ej: `r'(\2\3', urls)` --> esto reemplaza por el segundo y el tercer grupo

`findall` es similar a finditer pero devuelve una lista en vez de un iterable

`match` te dice si lo que hemos buscado está al principio del string (solo devuelve el primer resultado, no un iterable). No se usará mucho.

`search` busca en todo el string pero devuelve solo el primer resultado tambien.


Tambien se pueden añadir **FLAGS** a nuestros patrones:
`re.IGNORECASE` ignora mayusculas y minusculas (es lo mismo que [a-zA-Z])

---

## DATES

`import datetime`

_DATE (fecha)_: AAAA-MM-DD

:    Obtiene solo la fecha

_TIME (tiempo)_: HH:MM:SS.mmss

:    Se usa poco porque no tiene fecha

_DATETIME (fecha y hora)_: AAAA-MM-DD HH:MM:SS.mmss

:    Obtiene ambos atributos, es el que usaremos

Uso:

`datetime.date(y,m,d)`= para crear una fecha  
`datetime.date.today` = fecha de hoy

Si queremos solo un dato en concreto:

`*.today.year()` = solo el año  
`*.today.weekday()` = dia de la semana si Lunes es 0. Con `isoweekday()` Lunes es 1.

Los **timedeltas** son intervalos de tiempo en relación a una fecha.  
Con ello podemos:

* Restar o sumar un intervalo creado (Ej.: `tdelta = datetime.timedelta(days=7)`) a una fecha
* Calcular el intervalo entre dos fechas (si restamos o sumamos dos fechas, obtenemos un timedelta)

Ahora esto se lo podemos sumar por ejemplo a la fecha de hoy y ver de dentro de una semana.

Para la fecha actual hemos de tener en cuenta las zonas horarias.
`datetime.datetime.today()` o `datetime.datetime.now()` no tienen bien implementadas las zonas horarias.

Lo mejor es usar el paquete `pytz`: `datetime.datetime.now(tz=pytz.UTC)`

Si queremos convertirla: `hora.astimezone(pytz.timezone('Spain'))`

Para formatear horas en otro formato (ej: 23 de Julio de 2020)

`hora.strftime('formato')` pueder ver los distintos formatos en la documentación.
al reves (de string a hora) con `strptime()`

---
## TIME measure

```python
import time
start_time = time.time()
do_something()
print(“%s seconds” % (time.time() — start_time)) 
```

tambien se puede usar el magic method `%timeit` al inicio




## OS Module

* `os.getcwd()` = ver directorio activo
* `os.listdir()` = ls
* `os.walk(directory)` = recorre directorios (similar a `ls -R`)
* `os.stat(file)` = estadisticas de un archivo

* `os.chdir` = cambiar directorio
* `os.mkdir()` y `os.makedirs()` = crea directorios (sin y con subdirectorios)
* `os.rmdir()` y `os.removedirs()` = elimina "
* `os.remove(file)` = eliminar archivo
* `os.rename(old, new)` = renombrar

* `os.path.join()` = para juntar directorios (mejor que hacerlo sumando porque aqui se encarga solo de las barras \)
* `os.path.split()` = separa los subdirs y archivos
* `os.path.exists()` = checkea si existe
* `os.path.isfile/isdir()` = comprueba si es dir o archivo

* `shutil` module:
  * `shutil.copy2("source_file_path", "destination_directory_path")` — copiar archivos como `cp`
  * `shutil.move("source_file_path", "destination_directory_path")` — mover como `mv`

---

# REFERENCE

### == != is

`==` checks *equality*
dos latas de cocacola son iguales

`is` checks *identity*
dos latas de cocacola no son la misma lata (no son el mismo objeto que ocupa espacio en memoria

-----

### Subprocess

```python
import subprocess
p1 = subprocess.run()
	p1.args # te dice los argumentos que has dado
	p1.stdout # te da la salida si la has guardado
		# con capture_output = True
	p1.stderr # te da los errores (tambien .errorcode)
```


### str vs repr

* The goal of `str` is to be *readable* 
* the goal of `repr` is to be *unambiguous*

Vamos, que hay objetos que pueden no ser un string pero si tener una representación leible (que es la que buscamos con str), y con repr vemos lo que realmente es.

ej: "2016-10-22 12:13:43"
vs datetime.datetime(2016,10,22,12,13,43,tz=UTC)

### partition vs split

`partition` rompe solo una vez en el caracter dado y devuelve una tupla (previo, caracter, posterior)
`split` parte cada vez que se vea ese caracter

### Walrus-operator

Sirve para asignar (:=) y evaluar un booleano al mismo tiempo (asi nos ahorramos repetir una expresion)

```py
my_list = [1,2,3,4,5]
if (n := len(my_list)) > 3:    print(f"The list is too long with {n} elements")
```

---


# BIBLIOGRAFÍA

[Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)


```python

```
