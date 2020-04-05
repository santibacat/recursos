# CLASES
instancias de clase = objetos)
atributos y métodos
(=variables y funciones)

Tipos de métodos:

* Métodos de instancia = contienen self
* Métodos estáticos: no contienen self
* Metodos de clase

### ATRIBUTOS PRIVADOS
Los atributos privados solo pueden ser modificados por la clase y no por fuera (ni siquiera por la instancia de clase)..
Esto se hace haciendo __ delante del atributo al crearlo.
Ej `self.__password = password





### Variables de clase
Son parecidas, son variables que pertenecen a la clase y no a los objetos. Se ponen antes que el init de la clase y no necesitan instanciar el objeto antes.
Ej: pi en la clase circulo. Se podrá acceder con Circulo.pi y con circulo_uno.pi.

`object.__dict__` = te da un diccionario con todos los atributos de un objeto.

En python las variables de clase no son inmutables; por ello se le suele poner un guion bajo delante. _pi (por convencion).


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

### @staticmethods

Son metodos estaticos de una clase que no se pueden atribuir a un objeto real, pero que son útiles para la clase.
Le pertenecen a la clase, no a la instancia

Por ejemplo en la clase perro puede ser util llevar la cuenta de cuantos perros hay.
Para ello creamos una variable estatica numPerros (que es COMUN para todos los objetos de esa clase, si cambia una cambiará para los demás), y un método estático por ejemplo para mostrar el valor.

Dog.getNumDogs()

Además no hace falta crear una instancia para acceder a estos métodos estáticos.

(Si le llamamos dentro de la clase si podemos usar self.metodo() )



### NO PODER AÑADIR ATRIBUTOS
`object.__dict__` = te da un diccionario con todos los atributos de un objeto.

En una clase o en una instancia de clase podemos añadir los atributos que queramos (aunque no estén definidos).
Pero a veces estos atributos provocan errores; para ello podemos definir que una clase tenga permitido usar unos atributos definidos.

Para ello pondremos esto al principio de la clase:
`__slots__ = ['atributo1', 'atributo2']`

Esto además provoca que no podamos usar `__dict__`.
El atributo `__slots__` no se hereda (las subclases podrán establecer atributos que quieran).

## Herencia
Herencia multiple:

```
class Subclase(Clase1, Clase 2):
	return
```

Si un método es privado (__cazar) no puede ser accedido por las claes hijas
El orden de la herencia múltiple es de izquierda a dercha (para override de métodos)

### Métodos de clase
@classmethod
y cambiar self por cls

La diferencia es que los metodos estaticos pertenecen a la clase y nada más; los metodos de clase pertenecen a la clase y pueden usar los atributos y metodos publicos de las clases padre.

# ERRORES
try, except (as ..), raise, else, finally

Crear un error:

```
class myError(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)
```

