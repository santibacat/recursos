# Markdown basics

## <a name="part1"></a>Parte 1 Básica

Aqui pondré la formación básica de Markdown. Procede del tutorial de **MacDown**.

### Titulos 
**Negrita** con dos *  
*Cursiva* con un *  
_subrayado_ con un _

Salto de línea: doble retorno o dejar dos espacios al final del texto  
Sino aparece en la misma línea.

### Listas

* Lista 1 (con asterisco al principio)
* Lista 2
- Lista 3 (usar - hace lo mismo)
	- Lista indentada

1. Lista num1
	1. Lista indentada num (da igual el numero, el la ordena sola)
2. Lista num2

### Enlaces

`![texto](enlace "Titulo")`

[Enlaces sin imagen](http://www.google.com)  
![Enlaces con imagen](http://macdown.uranusjr.com/static/images/logo-160.png)


**Emails**: urls entre <>

<santibacat@gmail.com>

<http://www.google.com>


$~$


Puedes darle título al poner encima:
[Google](http://www.google.com "Google España")

`[Google](http://www.google.com "Google España")`

Podemos poner **enlaces** separados (para agruparlos en otro sitio):
[Hotmail][hotmail_web] (fijate doble corchete)

[hotmail_web]: http://www.hotmail.com "Outlook"

```
[Hotmail][hotmail_web] (fijate doble corchete)

[hotmail_web]: http://www.hotmail.com "Outlook"
```



### Citas

**Citas** con >, se pueden anidar a multiples niveles:

> Empezamos la cita
> > Anidamos un nivel
> > > Anidamos otro nivel
> Ahora volvemos al primero

### Código 
**Código** con `python code`

Si lo usamos doble es porque queremos usar corchetes en medio `` `(backticks)` ``

Si queremos hacer un bloque, solo debemos dar cuatro espacios o un tab:

    python code
    print("hola")

Tambien podemos hacer bloques con tres ' o ~ :
(especificar el lenguaje despues de las tres, como por ejemplo `` `python `` (y así pone colores del lenguaje).

```python
python code
import numpy as np
```

### Separación

Línea en blanco: `$~$`

**Linea divisoria:** tres * o -

***

---

$~$

## <a name="part2"></a>Parte 2: avanzada

Para darle un titulo a un apartado (para que vaya directamente un enlace) usamos:

`<a name="nombre-parte"></a>Titulo`

Tambien podemos usar headings ids de otra forma: `### Titulo {#custom-id}`

Y ahora para un enlace de ejemplo para ir arriba: [aqui](#part1) `[enlace](#parte)`


### Tablas


`| Left Aligned  | Center Aligned  | Right Aligned |`  
`|:------------- |:---------------:| -------------:|`  
`| col 3 is      | some wordy text |         $1600 |`  
`| col 2 is      | centered        |           $12 |`  
`| zebra stripes | are neat        |            $1 |`  

| Left Aligned  | Center Aligned  | Right Aligned |
|:------------- |:---------------:| -------------:|
| col 3 is      | some wordy text |         $1600 |
| col 2 is      | centered        |           $12 |
| zebra stripes | are neat        |            $1 |

### Inline Formatting

The following is a list of optional inline markups supported:

Option name         | Markup           | Result if enabled     |
--------------------|------------------|-----------------------|
Intra-word italic | So A\*maz\*ing   | So A<em>maz</em>ing
|
Intra-word bold     | So A\**maz\**ing   | So A**maz**ing
|
Strikethrough       | \~~Much wow\~~   | <del>Much wow</del>   |
Underline [^under]  | \_So doge\_      | <u>So doge</u>        |
Quote [^quote]      | \"Such editor\"  | <q>Such editor</q>    |
Highlight           | \==So good\==    | <mark>So good</mark>  |
Superscript         | hoge\^fuga\^    | hoge<sup>fuga</sup>   
|
Autolink            | http://t.co      | <http://t.co>         |
Footnotes           | [\^4] and [\^4]: | [^4] and footnote 4   |

[^4]: You don't have to use a number. Arbitrary things like `[^footy note4]` and `[^footy note4]:` will also work. But they will *render* as numbered footnotes. Also, no need to keep your footnotes in order, I will sort out the order for you so they appear in the same order they were referenced in the text body. You can even keep some footnotes near where you referenced them, and collect others at the bottom of the file in the traditional place for footnotes.  (Esto aparece al final porque es una nota).


### Ecuaciones LaTeX  
Se usan dos $ al inicio y al final:

$$T^{\mu\nu}=\begin{pmatrix}
\varepsilon&0&0&0\\
0&\varepsilon/3&0&0\\
0&0&\varepsilon/3&0\\
0&0&0&\varepsilon/3
\end{pmatrix},$$

integrals:

$$P_\omega={n_\omega\over 2}\hbar\omega\,{1+R\over 1-v^2}\int\limits_{-1}^{1}dx\,(x-v)|x-v|,$$

### Listas de tareas
* [x] Tarea completada  
* [ ] Tarea no completada


### Diccionarios

Palabra

:   Esta es la descripción


### Emojis

Se insertan entre dos : poniendo el nombre del emoji.

`:joy:` :joy:

Todos los emojis: [aqui](https://gist.github.com/rxaviers/7360908#file-gistfile1-md)


### Portada (Jekyll front-matter)
Hay que ponerla al inicio del archivo y usar tres ---:

```
---
title: "Macdown is my friend"
date: 2014-06-06 20:00:00
---
```

## <a name="part3"></a>Bibliografia

1. [Markdownguide](https://www.markdownguide.org/basic-syntax/)
2. [Macdown tutorial](https://github.com/MacDownApp/macdown/blob/master/MacDown/Resources/help.md)


# Utilidades Markdown

* Convertir markdown a jupyter (y mantener sincronizado): [jupytext](https://github.com/mwouts/jupytext):
  * Instalar: `conda install -c conda-forge jupytext`
  * Uso: en jupyter, `File > Jupytext` podemos darle a autoguardado
  * Desde comandos: `jupytext --to notebook --execute notebook.md    # convert notebook.md to an .ipynb file and run it`