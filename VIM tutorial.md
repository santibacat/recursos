# VIM
Resumen de <http://www.sromero.org/wiki/linux/aplicaciones/manual_vim>

## Bases
Tiene dos modos:

* Modo inserción: para meter el texto
* Modo comando: para ejecutar acciones.

Para pasar de inserción a comando: ESC  
Para pasar de comando a inserción: i, a

![tutorial vim](http://www.sromero.org/wiki/_media/linux/aplicaciones/vimman/vm_modos.png)

Podemos editar opciones por defecto en `.vimrc`:
`:set number` activa la numeración de líneas
`:syntax on` colorea palabras clave

Guardar los cambios en fichero:
`ZZ`o `:x!`

Al editar:  
`i` pasa al modo inserción  
`x` es como supr  
`X` es como <--  
`u` deshace ultima acción  
`CTRL+R` es redo  
`A` poner texto al final de la línea  
`o` crea una línea vacia  
`dd` borra la línea actual  
`D`borra desde la posición actual al fin de la línea

**Repetidores**:  
Consisten en repetir la tarea actual tantas veces como le pongas.  
`10dd` elimina 10 lineas.  
`10iHola<ENTER><ESC><ENTER>`inserta 10 veces hola en la pantalla cada vez en una línea.


# GITHUB

De <https://github.com/flowsta/github>

## Configuración
`git --version` ver versión instalada
`git config --global user.name "Tu nombre"`

## Crear un repositorio
```echo "# prueba" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/santibacat/prueba.git
git push -u origin master
```


Al usar ramas:  
`git push -u origin nueva-rama` envía archivos tras el commit  
`git branch -d rama-local`
`git push origin --delete rama-remota` elimina una rama remota



## Avanzado

`git config core.ignorecase false` = elimina el case-sensitive para nombres de archivos (ej: prueba.HTML y prueba.html se verán como dos)