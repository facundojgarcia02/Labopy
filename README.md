# Labopy:
Librería pensada para el análisis rápido de datos obtenidos en los laboratorios de la Licenciatura en Ciencias Físicas, para obtener conclusiones rápidas después de realizar las mediciones y poder discutir en el momento cualquier problema.

Esta construida utilizando Matplotlib, Scipy, Numpy y Sympy. Permite de manera rápida realizar ajustes no lineales en dos dimensiones, además de devolver valores de interés, como los test de bondad (Test F de Fisher, Test t de Student, Test chi cuadrado, y el parametro de correlación R cuadrado). 

Actualmente se puede elegir entre dos métodos para realizar el ajuste, Levenberg-Marquadt (que pertenece al grupo de 'Cuadrados mínimos no lineal'), y Regresion por Distancia Ortogonal (ODR en inglés, que pertenece al grupo de 'Cuadrados mínimos totales'). Proximamente se agregarán más posibilidades como dog-box, o incluso entregar al programa una función de costo personalizada.

Además, cuenta con una rápida visualizacióm de los residuos y su distribución, para corroborar posible "overfitting" o "underfitting" de nuestros datos.
