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




### Atributos PRIVADOS
Los atributos privados solo pueden ser modificados por la clase y no por fuera (ni siquiera por la instancia de clase).
Esto se hace haciendo __ delante del atributo al crearlo.
Ej `self.__password = password


### Properties

Sirven para trabajar con los atributos privados de una clase (ej self.__password) para no manipularla directamente:

```
@property
def password(self):
	return self.__password
	
@password.setter
def password(self, valor):
	self.__password = self.__generar_password(valor)
```


### Limitar atributos de una clase
`object.__dict__` = te da un diccionario con todos los atributos de un objeto.

En una clase o en una instancia de clase podemos añadir los atributos que queramos (aunque no estén definidos).
Pero a veces estos atributos provocan errores; para ello podemos definir que una clase tenga permitido usar unos atributos definidos.

Para ello pondremos esto al principio de la clase:
`__slots__ = ['atributo1', 'atributo2']`

Esto además provoca que no podamos usar `__dict__`.
El atributo `__slots__` no se hereda (las subclases podrán establecer atributos que quieran).



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


### Métodos ESTÁTICOS

> @staticmethod


No pasan ningun argumento al método (ni self ni cls) a diferencia de los anteriores. Son como funciones normales pero con alguna conexión con la clase.

No se pueden atribuir a un objeto real (ni a la clase), pero son útiles para la clase.
Le pertenecen a la clase, no a la instancia.

Ej: un método que calcula si es festivo o no para los empleados (no usa nada de la clase).





------



## Herencia
Herencia multiple:

```
class Subclase(Clase1, Clase 2):
	return
```

Si un método es privado (__cazar) no puede ser accedido por las claes hijas
El orden de la herencia múltiple es de izquierda a dercha (para override de métodos)



-------
__NO VA AQUI__


# ERRORES
try, except (as ..), raise, else, finally

Crear un error:

```
class myError(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)
```

