# CS2702: Proyecto 3 - Face Recognition

Desarrollado por: **Efraín Córdova**, **Miguel Lama** y **Diego Paredes**.

## Demo en video

## ¿Cómo se usa el aplicativo?

## Librerías utilizadas

### Face Recognition

(*Documentación en: https://face-recognition.readthedocs.io*)

Dada una imagen en disco, la librería de face recognition identiica los rostros que se encuentran en dicha imagen (*load_image_file*). Luego, genera un vector de 128 dimensiones que representa una codificación de alguno de los rostros identificados (*face_encodings*). Para este proyecto, llamaremos **codificación** al vector de 128 dimensiones que representa al rostro más reconocible de la imagen. Guardaremos la colección de codificaciones en disco, en `data/raw.csv`. La codificación puede usarse para comparar distintas imágenes, como puede apreciarse en el siguiente gráfico:

![](https://cdn.discordapp.com/attachments/917173840377937960/917176643620065341/fr.png)

### Rtre

*(Documentación en: https://rtree.readthedocs.io)*

Las codificaciones son insertadas en una estructura Rtree que se almacena en disco. Utilizaremos índices con distintos números de elementos para poder evaluar el rendimiento de los algoritmos de búsqueda. La construcción de estos índices se realiza en `build_idxs.py`

![](https://cdn.discordapp.com/attachments/917173840377937960/917185308867571752/built_time.png)

Este árbol divide el espacio en regiones locales (MBRs), por lo que las consultas sobre la estructura accederán solo a aquellas regiones que sean espacialmente próximas al parámetro de búsqueda. Usaremos la función **nearest()** para comparar el rendimiento del algoritmo de búsqueda de k vecionos más cercanos que fue implementado, además de la función **interesection()** como subrutina en la búsqueda por rango.

## Estrategia de mitigación de dimensionalidad

Para evitar pérdidas en el rendimiento, se decidió usar sólo aquellas características de la codificación que tengan una significancia mayor o igual al 0.93 utilizando la líbreria de PCA (Principal Component Analysis). Esta reducción implica una pérdida de precisión en el reconocimiento de imágenes. Experimentalmente, se comprobó que la eficacia de la líbreria face_recognition sobre el dataset es ahora de un **98.39%**, frente al **99.38%** de precisión que se obtiene sin la reducción. Esta experimentación se realizó en `preprocessing/measure_reduction.py`

![](https://cdn.discordapp.com/attachments/707425093256609834/917188925066457128/measure1.png)
![](https://cdn.discordapp.com/attachments/707425093256609834/917188925284573214/measure2.png)
![](https://cdn.discordapp.com/attachments/707425093256609834/917188925498486844/measure3.png)
![](https://cdn.discordapp.com/attachments/707425093256609834/917188925670457425/measure4.png)

Con esta reducción, la codificación es ahora un vector de 55 dimensiones. Esta nueva codificación reemplaza la ya existente y es la que se usa en todo el proyecto. Al igual que la codificación original, esta se almacena en disco. El proceso de escritura de codificaciones quedaría de la siguiente forma: Se codifican las imagenes de la base de datos y se escriben en disco (`preprocessing/raw_writter.py`). Luego, esas codificaciones son reducidas con PCA  (`preprocessing/dim_reductor.py`) y son guardadas en `data/reduced.csv`. Es esta colección la que posteriormente será insertada en el Rtree.

![](https://cdn.discordapp.com/attachments/917173840377937960/917192918316507166/red_process.png)

## Algoritmos de búsqueda (KNN y Range)

*(Los códigos los puede encontrar en `search_algorithms.py`)*
***
![](https://cdn.discordapp.com/attachments/917173840377937960/917265045103214632/unknown.png)

Se desarrolló el algoritmo haciendo uso de la cola de prioridad implementada en `heapq`, manteniendo la máxima capacidad de la cola en k. 

![](https://cdn.discordapp.com/attachments/917173840377937960/917249000791494656/knn_search.png)

Este algoritmo construye un heap en *O(nlgk)*, ya que realiza un pop y un push luego de insertar los k primeros elementos. Luego, realiza un heapsort del heap en *O(klgk)*. Adicionalmente invertimos el arreglo en *O(k)* para poder comparar los outputs del rtree. La búsqueda, que comprende la construcción del heap y el heapsort, se realiza entonces en *O((n+k)lgk)*.
***

![](https://cdn.discordapp.com/attachments/917173840377937960/917264609071730758/unknown.png)

Para este algoritmo, como se indicó anteriormente, se hace uso de la función **intersection()** para encontrar aquellos MBRs que tienen alguna intersección con el aréa dentro del rango de búsqueda. Luego, los primeros candidatos son todos los puntos dentro de esos MBRs. Al filtrar a aquellos candidatos que se encuentran en el rango de búsqueda se tiene la respuesta final a la consulta.

![](https://cdn.discordapp.com/attachments/917173840377937960/917252060242645083/range_search.png)

La búsqueda de los primeros candidatos se realiza en *O(lgn)*. Luego, el segundo filtro de candidatos se realiza en tiempo lineal con respecto a la cantidad de candidatos recibidos, que puede llegar a ser O(n). Adicionalmente realizamos un ordenamiento en *O(nlgn)*. La búsqueda, que comprende ambos filtros de candidatos, se realiza entonces en *O(nlgn)*. 

## Experimentación y Benchmarks

### Cuado comparativo de KNN

Se lleno el cuadro comparativo entre KNN-Rtree y KNN-Secuencial usando colecciones de datos con distinto número de elementos (ver `benchmarks.py`). 

<p align="center">
  <img src="https://cdn.discordapp.com/attachments/917173840377937960/917269304850935848/bd2_p3_helper_images.png" />
</p>

### Elecciones de radio 

Se escogieron los radios: **0.6, 0.7275, 0.792 y 0.85496**. En primer lugar, 0.6 es la tolerancia por defecto que emplea la librería de face_recognition para estimar cuando dos rostros corresponden a la misma persona, así que consideramos que tomar este valor como radio para las mediciones es acertado. Luego, los últimos 3 valores corresponden respectivamente a los cuantiles comprobados experimentalmente en 3 iteraciones:
<p align="center">
  <img src="https://cdn.discordapp.com/attachments/917173840377937960/917261420587008020/unknown.png" />
</p>
<p align="center">
  <img src="https://cdn.discordapp.com/attachments/917173840377937960/917261848154365953/unknown.png" />
</p>

<p align="center">
  <img src="https://cdn.discordapp.com/attachments/917173840377937960/917262290976399400/unknown.png" />
</p>



### Benchmarks de búsqueda de rango

Se realizó un segundo cuadro comparativo para esta búsqueda, donde se consideraron el número de elementos devueltos y la media de los tiempos de ejecución, en 4 iteraciones. Se consideró como imagen de búsqueda a `Aaron Eckhart`, del cual se tiene una única imagen suya en la colección.

<p align="center">
  <img src="https://cdn.discordapp.com/attachments/917173840377937960/917264005863735336/benchmarks_range.png" />
</p>

## Glosario de directorios

 - **data**: Directorio que contiene dos csv: raw.csv (codificaciones originales en 128D) y reduced.csv (codificaciones reducidas en 55D).
 - **indexes**: Directorio que contiene 9 índices con distinta cantidad de elementos, uno de ellos contiene a toda la colección y es el que se usa por defecto.
 - **LFW_images**: Directorio que contiene las 13233 imágenes de la colección proporcionada.

## Otras dependencias

 - Heapq
 - PrettyTable
 - PCA
 - Pandas
 - Numpy
 - Mathplotlib
