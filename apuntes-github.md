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