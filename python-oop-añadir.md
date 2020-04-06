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
Se empiezan con `__`

`__repr__`: unambiguous representation  (solo para que la vean otros desarrollaodres).
`__str__`: readable representation  (para el usuario final).
> Por defecto se usa str>repr cuando hacemos un print

Otros:
`__add, sub, mul, matmul, __` = para hacer cálculos con clases
Para el cálculo se hace usando other para el otro argumento:
```
def __add__(self, other):
	return (self.pay + other.pay)
```



`__len__` = calcular la longitud


