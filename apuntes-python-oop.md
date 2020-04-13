# CLASES

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


## ATRIBUTOS

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
Los atributos privados solo pueden ser modificados por la clase y no por fuera (ni siquiera por la instancia de clase).
Esto se hace haciendo __ delante del atributo al crearlo.
Ej `self.__password = password



### Limitar atributos de una clase
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

```
@property
def password(self):
	return self.__password
	
@password.setter
def password(self, valor):
	self.__password = self.__generar_password(valor)
	
@password.deleter
def delete_password(self):
	self.__password = None
	print("Password deleted")
```

---------







## MÉTODOS


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

La `cls`del final se refiere a la clase, y es lo mismo que si pusieramos `Employee`. De esta forma estamos creando la clase con un método:  
`employee1 = Employee.from_string('Santi-Bacat')`


### Static methods (métodos estáticos)

> @staticmethod


No pasan ningun argumento al método (ni self ni cls) a diferencia de los anteriores. Son como funciones normales pero con alguna conexión con la clase.

No se pueden atribuir a un objeto real (ni a la clase), pero son útiles para la clase.
Le pertenecen a la clase, no a la instancia.

Ej: un método que calcula si es festivo o no para los empleados (no usa nada de la clase).





------



## Herencia

Nos permite obtener las características de la clase padre, superponerlas o añadir nuevas.

Primero busca el init en la subclase y luego va a la clase padre.

* Si no lo encuentra, ejecuta el init de la clase padre.
* Si lo encuentra, lo ejecuta aquí. Si hay argumentos repetidos, lo pasamos a la clase padre (para mantener código reproducible). `super().__init__(arg1repetido, arg2repetido)`
  * Tambien podríamos poner el nombre de la clase, pero esto se usa más en herencias múltiples:  `Employee.__init__(arg1repetido, arg2repetido)`



Podemos usar la funcion help para ver todo sobre la clase (la subherencia, atributos heredados...).
Podemos usar `isinstance()` para ver un objeto es instancia de una clase: `isinstance(manager1, Developer)`
Podemos usar `issubclass()` para ver si una clase es subclase de otra (<--): `issubclass(Manager, Employee)` es True.


### Herencia multiple:

```
class Subclase(Clase1, Clase 2):
	return
```

Si un método es privado (__cazar) no puede ser accedido por las claes hijas
El orden de la herencia múltiple es de izquierda a dercha (para override de métodos)



## Magic Methods
Son `__methods__` y sirven para establecer acciones por defecto.

`__repr__`: unambiguous representation  (solo para que la vean otros desarrollaodres).
`__str__`: readable representation  (para el usuario final).
> Por defecto se usa str>repr cuando hacemos un print

`__add, sub, mul, matmul, __` = para hacer cálculos con clases
Para el cálculo se hace usando other para el otro argumento:

```
def __add__(self, other):
	return (self.pay + other.pay)
```



`__len__` = calcular la longitud




-------
__NO VA AQUI__


# RESTO DE PYTHON


### ERRORES
try, except (as ..), raise, else, finally

* `try` lo que se intenta ejecutar
* `except ErrorType` lo que ejecuta si hay un error de un tipo
* `raise` para activar un error manualmente
* `else` se ejecuta si try no da error
* `finally` código que se ejecuta al final haya dado error o no



Crear un error personalizado:

```
class myError(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)
```





---------
### if name == main

```
if __name__ == `__main__`:
	main()
```

Sirve para verificar si un archivo está siendo ejecutado directamente (llamando a ese archivo) o es importado.

* Si se llama directamente, lo que hay dentro de `main()`se ejecuta gracias a esta línea. Además tambien se ejecuta todo lo que esté fuera.
* Si se importa ese archivo a otro (`import primerpaquete`) se ejecutaría todo lo que está fuera de `main()`. Por ello, si hay algo que no queremos que se ejecute al importarlo, lo ponemos dentro de `main()`.
	* Ej: si solo queremos importar paquetes o algo para reutilizar código.
* Si queremos ejecutar este código en el otro archivo lo podremos llamar usando `primerpaquete.main()`.


-----

### == + is

== checkea igualdad
dos latas de cocacola son iguales

is checkea identidad
dos latas de cocacola no son la misma lata (no son el mismo objeto que ocupa espacio en memoria

-----
### Subprocess

```
import subprocess
p1 = subprocess.run()
	# p1.args = te dice los argumentos que has dado
	# p1.stdout = te da la salida si la has guardado
		# con capture_output = True
	# p1.stderr = te da los errores (tambien .errorcode)
	

```


---
### Variable scope

Es el alcance de cada variable en python

*LEGB* es en orden en que python busca:

* Local
* Enclosing
* Global
* Built-in

_Enclosing-scope_ se refiere a que todo lo interior incluye a lo exterior, pero no al revés.


Podemos especificar que dentro de una funcion se use un valor global y no un valor local de dentro de la funcion:

```
x = 'outside x'

def test():
	global x
	x = 'inside x'

print(x) # = 'inside x'
```

Lo mismo podemos hacer en funciones anidadas, cambiar un valor más externo pero sin afectar al valor local

```
def outer():
	x = 'outer x'
	def inner():
		nonlocal x
		x = 'inner x'
	inner()
outer() # dará 'inner x'
```


_Built-in_ se refiere a nombres globales de python.
Tenemos que tener cuidado porque podemos sobreescribirlas con nuestras propias funciones y no da error.

```
import builtins
print(dir(builtins))
```

FUNCIONES

Para dejarlas vacías

```
def hola():
	pass
```

cuando le pasamos *args y **kwargs a una función tiene que ser en ese orden.
Además si se las pasamos empaquetadas no funcionará bien. Ej.
courses = ['math', 'science']
info = {'name': 'John'; 'age': 28}

Si al pasarlo a la funcion student_info(courses, info) lo pasamos sin nada, no desempaquetará bien y lo tomará todo como args.

hay que desempaquetarlo como args y kwargs
`student_info(*courses,**info)`




---
os module

* os.chdir = cambiar directorio
* os.getcwd() = ver directorio activo
* os.listdir() = ls
* os.mkdir() y os.makedirs() = crea directorios (sin y con subdirectorios)
* os.rmdir() y os.removedirs() = elimina "
* os.rename(old, new) = renombrar
* os.stat(file) = estadisticas de un archivo
* os.walk = recorre directorios 
* os.path.join() = para juntar directorios (mejor que hacerlo sumando porque aqui se encarga solo de las barras \)
* os.path.split() = separa los subdirs y archivos
* os. path.exists() = checkea si existe
* os.path.isfile/isdir () = comprueba si es dir o archivo



TRABAJAR CON FECHAS

### Datetime
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

### BE PYTHONIC

**Duck typing**

Se refiere a no tener en cuenta el tipo de objeto si puede hacer lo que se le permite.

> Si anda como un pato y vuela como un pato, se le trata como a un pato.



**EAFP (*easier to ask forgiveness than permission*)**

Se refiere a que en otros lenguajes vamos anidando condicionales para hacer comprobaciones antes de ejecutar (*LBYL = look before you leak*), y en python es mejor dejar ejecutar y usar try/except para ver los errores.


---

### FORMATTING STRINGS

La forma antigua es con `.format()`.

`'My name is {} {}'.format(name, surname)`

Se pueden poner números en los corchetes por si hay valores repetidos:
`I'm {0}. {0} {1}'.format(name, surname)`

#### f-strings

A partir de python 3.6 

New: `f'My name is {name} {surname}`

Es preferible usar double_quotes para que no afecte a las variables que haya dentro:  
`f"My name is {person['name']}"`

Para formatear números se usan _dos puntos_: `{n:02.3f}`

Para formatear fechas igual: `{birthday:%B %d %Y}`


### Generators

Es como hacer una lista pero usando `yield`. Mejor porque usa menos memoria (solo genera uno a la vez).
Para obtener el siguiente resultado `next`.
Cuando termina todos ya no se puede iterar más (da `StopIteration`).

Cuando hacemos un for con una list comprehension en el fondo estamos haciendo un generador; para ello debemos usar `()` paréntesis.
`(x*x for x in [1,2,3,4,5])`

### Decorators

Primero habla de los **closures** que yo no habia oido hablar; creo que son funciones anidadas en la que se guarda la ejecución pero no se ejecuta:

```
def outer_function(msg):
	def inner_function():
		print(msg)
	return inner_function

hi_func = outer_function('Hi')
```

De esta forma se crea la función con los parametros pero aun no se ejecuta.

Esta es la base de funcionamiento de los decoradores, que modifican la ejecución de una función (wrapped):

```py
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

```py
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

#### DECORADORES CON ARGUMENTOS

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


### namedTuples

Son como tuplas pero más entendibles. Y tienen las ventajas de las tuplas (inmutables y mas rápidas que los diccionarios).

Ej: una tupla con colores RGB, para identificar correctamente cual es cada color.

```python
from collections import namedtuple
# namedtuple('nombre', ['args'])

Color = namedtuple('Color', ['red','green', 'blue'])
new_color = Color(55,155,255)
new_color.blue # podemos acceder individualmente
```



### COMPREHENSIONS

```python
[n^2 for n in nums if n%2 == 0]
[(letter,num) for letter in 'abcd' for num in range(4)]
```

Tambien se pueden hacer con diccionarios y sets:

```python
{name:hero for name, hero in zip(names, heros)if name != 'Peter}
```

Los sets son como listas pero con valores unicos

```python
{n for n in nums}
```

Los generadores son muy similares a las comprehensions.

Recuerda que `zip` crea tuplas con el primer indice de cada lista, luego el segundo, y así hasta el final.

### SORT OBJECTS

Podemos usar `sorted()` que crea nueva variable o el método `list.sort()` para una ya creada.

`reverse=True` la ordena al revés
`key=` es una función que diga *cómo* ordenar. Puede ser un lambda tipo `key = lambda e: e.salary` o un getter `key = attrgetter('age')`.


### RANDOM

`import random`

`random.random()`= valor entre 0 y 1
`random.uniform(inicio, final)` = devuelve float
`random.randint(inicio, final)` = devuelve enteros
`random.choice(list)` = devuelve un valor aleatorio de entre una lista
`random.choices(list, k=veces)` = devuelve una lista de `k` valores aleatorios de entre una lista.
* `weights = [50, 25, 25]` para ponderar el peso de cada elemento original de la lista 

Valores unicos (evitar repeticion), no se usa choices sino `random.sample(list, k=)`.


### REGULAR EXPRESSIONS

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

Además de `finditer`tambien podemos usar `sub` que sustituye el patrón encontrado por un string que le pasemos:
`pattern.sub(sub_pattern, text)`
Ej: `r'(\2\3', urls)` --> esto reemplaza por el segundo y el tercer grupo

`findall` es similar a finditer pero (no se la diferencia, creo que es )

`match`te dice si lo que hemos buscado está al principio del string (solo devuelve el primer resultado, no un iterable). No se usará mucho.

`search` busca en todo el string pero devuelve solo el primer resultado tambien.


Tambien se pueden añadir **FLAGS** a nuestros patrones:
`re.IGNORECASE` ignora mayusculas y minusculas (es lo mismo que [a-zA-Z])


### str vs repr

The goal of str is to be readable and the goal of repr is to be unambiguous

Vamos, que hay objetos que pueden no ser un string pero si tener una representación leible (que es la que buscamos con str), y con repr vemos lo que realmente es.

ej: "2016-10-22 12:13:43"
vs datetime.datetime(2016,10,22,12,13,43,tz=UTC)

### partition vs split

partition rompe solo una vez en el caracter dado y devuelve una tupla (previo, caracter, psoterior)
split parte cada vez que se vea ese caracter
