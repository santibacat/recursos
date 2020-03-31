# GIT
## Comandos básicos
Crear un repositorio nuevo:  

```
# Crear carpeta e ir a ella
mkdir repositorio
cd repositorio 

# Crear y subir repositorio
git init
touch README.md
git add . # añade todo, tambien git add *
git commit -m "Nombre del commit"
git push -u origin master

```

Clonar un repositorio:  
`git clone /path/repo.git`


__Información básica:__

* Estado de la sincronización: `git status`
* Log de los commits: `git log`




## Explicación básica de Git

Tenemos tres zonas locales y una remota:

* WD (Working directory): archivos actuales
* Index (Staging area): area de trabajo
* HEAD (Local repo)
* MASTER (Remote repo, Github)

![https://rogerdudler.github.io/git-guide](https://rogerdudler.github.io/git-guide/img/trees.png)
![](https://cdn-media-1.freecodecamp.org/images/1*iL2J8k4ygQlg3xriKGimbQ.png)

__ADD & COMMIT__

* Para añadir todos los archivos: `git add *`  
* Para añadir uno solo: `git add <archivo>`
* Para hacerlo interactivo: `git add -i`

Para hacer commit: `git commit -m "Nombre"`
Para editar un commit (pero que se guarden los cambios dentro del mismo): `git commit --amend -m "Fix"`

__REPOSITORIO REMOTO__

Configurar repositorio remoto (si no está ya configurado):  
`git remote add origin <server>`

Ver repositorios remotos:
`git remove -v`

Para enviar los cambios:  
`git push origin <rama>`

Generalmente usaremos `git push origin master`


## Ramas

![https://rogerdudler.github.io/git-guide](https://rogerdudler.github.io/git-guide/img/branches.png)

Para ver las ramas: `git branch`

Para cambiar de rama: `git checkout <rama>`

Modificar ramas:

```
git checkout -b rama1 # crear
git branch -d rama1 # eliminar rama local
git push origin --delete rama-remota
```

Enviar una rama al repositorio remoto: `git push origin <rama>`

## Actualizar repo

Se hace en dos partes:

* `git fetch` sirve para traer archivos del repositorio remoto al local (pero no al WD)
* `git merge <rama>`mezcla el contenido del repo local con el del WD

Estos dos comandos equivalen a hacer `git pull`

Esto solo funciona si no hay confictos ('auto-merge'). Cuando hay conflictos hay que hacerlo a mano, editando los archivos.  
Podemos ver las diferencias entre archivos:  
`git diff <source-branch> <target-branch>`

## Tagging

Sirve sobre todo para añadir nombres de versión a los commits (ej: v1.2).

Añadir versión al ultimo commit:
`git tag 1.0.0`

Añadir versión aun commit concreto:

```
git log # para ver la id del commit
git tag 1.0.0 <idcommit>
```

Eliminar una etiqueta: `git tag -d v2`

## Log

Comandos avanzados del log:

* Ver solo los commits de un autor: `git log --author=santi`
* Ver un log comprimido: `git log --pretty=oneline`
* Ver logtree: `git log --graph --oneline --decorate --all`
* Ver solo archivos cambiados: `git log --name-status`

## Cambios (diff)

* Ver cambios en esta versión: `git diff`
* Ver cambios entre un archivo modificado y el local: `git diff file.ext`
* Ver cambios entre versiones: `git diff v1-v2`

## Restaurar cambios

Eliminar ultimo commit del repo local (restaurar commit anterior):

* Volver un archivo al original: `git checkout -- <nombre>`
* Volver todos los archivos al original: `git checkout .`
* Descartar todos los cambios locales: 

```
git fetch origin
git reset --hard HEAD # al ultimo commit
git reset --hard origin/master # al repo remoto

```

Si hemos realizado commit y push:
`git revert` (busca como)


## Forks

Actualizar un repositorio forkeado:

* Sin integrar los cambios: `git fetch upstream`
* Integrando los cambios con la versión local: `git pull upstream master`




## Configuración avanzadda

### Comandos avanzados
* Ver versión instalada: `git --version`
* Configurar datos de usuario: `git config --global user.name "Tu nombre"`
* Eliminar case sensitive de los archivos: `git config core.ignorecase false`
	* Se verán dos archivos para readme.md y README.md
* More to come


Git usa `vi`como editor principal (se puede cambiar a `vim` o `nano`)

* `:q` salir
* `:wq` salir y guardar



### Gitignore

En `.gitignore` añadimos archivos que no quedamos sincronizar:

* `.DS_store`
* `.ipynb_checkpoints/` carpetas

Si queremos eliminar un archivo remoto al que ya hemos hecho push antes de añadirlo a .gitignore: `git rm -r --cached <carpeta>`

## Otros

* Interfaz gráfica git: `gitk`

## Bibliografía

* [Git, the simple guide](https://rogerdudler.github.io/git-guide/) by Roger Dudler
* [Learn Git basics under 10 minutes](https://www.freecodecamp.org/news/learn-the-basics-of-git-in-under-10-minutes-da548267cc91/) de FreeCodeCamp
* [Tutorial git](http://flowsta.github.io/github/) de Flowsta
* [Curso Youtube Introducción a GIT](https://www.youtube.com/watch?v=zH3I1DZNovk) de CodigoFacilito