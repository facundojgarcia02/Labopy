#Importaciones
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def gaussian(x: float,a: float,b: float,c: float) -> float:
	"""
	Función gaussiana a*np.exp(-(x-b)**2/(2*c**2)). Se utiliza para crear la curva del histograma.
	
	Argumentos:
	-	x: np.darray o float.
	-	a: np.darray o float.
	-	b: np.darray o float.
	-	c: np.darray o float.

	Retorna: 
		np.darray o float.

	Notas: 
	1.	Si se utilizan dos o más ndarray, tienen que ser del mismo tamaño.
	"""

	return a*np.exp(-(x-b)**2/(2*c**2))

class Expresion():

	def __init__(self, symbol_list: list = None, sympy_expr: sp.Basic = None):
		"""
		Se utiliza en los ajustes y graficos, además de permitir propagar el error de manera rápida.

		Parámetros:
		-	symbol_list: list -> Contiene la expresión que vayamos a utilizar.
		-	sympy_expr: sp.Basic -> Expresión de Sympy.

		Notas: 
		1.	Es importante el orden de los símbolos, ya que habrá que respetarlos a la hora de trabajar con los distintos métodos de la clase.
		"""

		self.save_info(symbol_list, sympy_expr)

	def check_errors(self, symbol_list: list = None, sympy_expr: sp.Basic = None):
		"""
		Método que chequea posibles errores al crear el objeto o modificar la información que guarda.

		Parámetros:
		-	symbol_list: list -> Contiene la expresión que vayamos a utilizar.
		-	sympy_expr: sp.Basic -> Expresión de Sympy.
		"""

		#------------------------------------------------
		#Chequeamos los argumentos que le estamos pasando.
		if type(symbol_list) != list:
			raise TypeError("Se requiere la lista de símbolos. Uso: Expresion(symbol_list,sympy_expr)")

		elif sympy_expr == None:
			raise TypeError("Se requiere la expresión creada por Sympy. Uso: Expresion(symbol_list,sympy_expr)")

		#----------------------------------------------
		#Chequeamos que no falte ni sobre ningún símbolo.
		sympy_expr_symbols = sympy_expr.atoms(sp.Symbol)
		
		#Sobran símbolos en la lista.
		for symbol in symbol_list:
			if not (symbol in sympy_expr_symbols):
				raise ExpresionError(f"Símbolo no encontrado en la expresión de Sympy entregada: {sp.latex(symbol)}. Chequeá que en el argumento 'symbol_list' no sobre ningun símbolo.")
		
		#Faltan símbolos en la lista.
		if set(sympy_expr_symbols) != set(symbol_list):
			print(set(sympy_expr_symbols))
			print(set(symbol_list))
			raise ExpresionError(f"Símbolos faltantes en 'symbol_list': {sympy_expr_symbols.difference_update(symbol_list)}")

	def save_info(self, symbol_list: list = None, sympy_expr: sp.Basic = None):
		"""
		Método que guarda la información en el objeto.

		Parámetros:
		-	symbol_list: list -> Contiene la expresión que vayamos a utilizar.
		-	sympy_expr: sp.Basic -> Expresión de Sympy.
		"""

		#Evitamos errores de antemano:
		self.check_errors(symbol_list,sympy_expr)

		#Funcionamiento propio de la clase.

		self.symbol_list = symbol_list
		self.expresion = sympy_expr

		#Creamos los sigmas para cada variable con error que vamos a pasar.
		self.error_symbols_list = []
		for _symbol in symbol_list:
			latex_symbol_expr = sp.latex(_symbol)
			self.error_symbols_list.append(sp.Symbol(r"\sigma_{" + latex_symbol_expr +"}"))
			#Nota: Los símbolos del error estan ordenados de la misma forma que los símbolos pasados.

		#Creamos la expresión del error:
		self.error_expresion = sp.N(0)
		for index, _symbol in enumerate(symbol_list):
			self.error_expresion += sympy_expr.diff(_symbol)**2 * self.error_symbols_list[index]**2
		self.error_expresion = sp.sqrt(self.error_expresion)

		#Creamos expresiones que se pueden evaluar.
		self.evaluate_expresion = sp.lambdify(self.symbol_list,self.expresion)

		#Nota: Vamos a tener que pasarle a 'evaluate_error_expresion' primero los valores de cada magnitud, y despues el error de cada una, en el mismo orden.
		#Ejemplo para a*x+b si la lista era [x,a,b], [x,a,b,sigma_x,sigma_a,sigma_b]
		self.evaluate_error_expresion = sp.lambdify([*self.symbol_list,*self.error_symbols_list],self.error_expresion)

	#TERMINAR
	def propagate_error(self,parameter_values: list, error_values: list):
		"""
		Método que calcula la propagación del error a partir de los valores para cada parámetro y su error asociado.

		Parámetros:
		-	parameter_values: list -> Lista con los valores (puede ser un valor o un np.ndarray por variable [Si son np.ndarray se requiere que sean del mismo tamaño]) de cada variable, ordenados según se pasaron los símbolos al crear el objeto.
		-	error_values: list -> Lista con los valores (puede ser un valor o un np.ndarray por variable [Si son np.ndarray se requiere que sean del mismo tamaño]) del error de cada variable, ordenados igual que los símbolos que se pasaron al crear el objeto.
		
		Retorna:
		-	'float' o 'np.ndarray' con los valores de error propagados.
		"""
		pass

class Ajuste():

	def __init__(self, x: list, y: list, expr: sp.Basic, y_err: list, x_err: list = 0):
		"""
		Objeto que guarda la información que vamos a ajustar. Con este podemos hacer el gráfico y los test de bondad necesarios para corroborar si nuestro ajuste es bueno.

		Parámetros:
		-	x: array-like -> Valores de la variable independiente.
		-	y: array-like -> Valores de la variable dependiente.
		-	expr: labopy.Expresion -> Expresión que almacena la forma funcional a ajustar.
		-	y_err: array-like o int -> Cantidad de incerteza en la variable dependiente.
		-	x_err: array-like o int (opcional, Default = 0) -> Cantidad de incerteza en la variable independiente. Si es despreciable se toma 0 automaticamente.

		Notas: 
		1.	El primer símbolo que hayamos pasado al crear la expresión sera el utilizado como variable dependiente.
		"""

		#Guardar información, además, ya ajusta automaticamente con el parámetro 'adjust'.
		self.save_info(x,y,expr,y_err,x_err,adjust = True)

	def save_info(self,x: list,y: list,expr: sp.Basic,y_err: list,x_err: list,adjust: bool = True):
		"""
		Método que guarda la información necesaria para ajustar en el objeto.

		Parámetros:
		-	x: array-like -> Valores de la variable independiente.
		-	y: array-like -> Valores de la variable dependiente.
		-	expr: labopy.Expresion -> Expresión que almacena la forma funcional a ajustar.
		-	y_err: array-like o int -> Cantidad de incerteza en la variable dependiente.
		-	x_err: array-like o int (opcional, Default = 0) -> Cantidad de incerteza en la variable independiente. Si es despreciable se toma 0 automaticamente.
		"""

		#Guardamos la informacion que le pasamos en el objeto.
		self.x = np.array(x)
		self.y = np.array(y)

		#Si pasamos un solo valor, 
		if type(y_err) in [int,float]:
			self.y_err = np.ones_like(y)*y_err
		elif type(y_err) in [list,np.ndarray]:
			self.y_err = np.array(y_err)
		else:
			raise TypeError("Se requiere una lista de valores para el error en y, o en su defecto, un solo valor pasado como 'float' o 'int'")

		#Guardamos la información que necesitamos de la expresión.
		self.expr = expr
		self.test_function = self.expr.evaluate_expresion

		self.adjust()

	def adjust(self):
		"""
		Método que se encarga de realizar el ajuste y guardar en el objeto la información relevante de este.
		"""

		self.popt, self.pcov = curve_fit(self.test_function,self.x, self.y,sigma = self.y_err, absolute_sigma = True)
		self.perr = np.sqrt(np.diag(self.pcov))

		self.model_y = self.test_function(self.x,*self.popt)

		self.TSS = sum((self.y-np.mean(self.y))**2)
		self.RSS = sum((self.y-self.model_y)**2)
		self.ESS = sum((self.model_y-np.mean(self.y))**2)

	def full_analisis(self):
		"""
		Ejecuta todos los test de bondad necesarios, y crea el gráfico con residuos e histograma para chequear distribución de los últimos.
		"""

		print("Mostrando el análisis completo del ajuste:")

		self.test_chi()
		self.test_f()
		self.test_t()
		self.test_r()
		self.make_graph()

	def test_r(self):
		"""
		Realiza el test 'R^2'.
		Muestra en pantalla la información, no retorna nada.
		"""

		print("\n\n------- COEFICIENTE DE DETERMINACIÓN -------\n")

		r2 = r2_score(self.y,self.model_y)
		print(f"Coeficiente de determinación R^2: {r2}")

	def test_t(self):
		"""
		Realiza el test 't' de Student.
		Muestra en pantalla la información, no retorna nada.
		"""

		print("\n\n------- TEST 't' DE STUDENT -------\n")

		t_arr = [abs(popt_i/perr_i) for popt_i,perr_i in zip(self.popt,self.perr)]
		p_arr = [stats.t.sf(t_i,len(self.x)-len(self.popt))*2 for t_i in t_arr]

		for i in range(len(self.popt)):
			print("\n") #Breakline
			print(f'{sp.latex(self.expr.symbol_list[i+1])} = ' + str(self.popt[i]) +  u' \u00B1 ' + str(self.perr[i]) )
			print('p-valor del t: ' + str(p_arr[i]))
			if p_arr[i]<0.05:
				print('El parámetro es estadísticamente significativo.')
			else:
				print('El parámetro no es estadísticamente significativo.')

	def test_chi(self):
		"""
		Realiza el test 'Χ^2'.
		Muestra en pantalla la información, no retorna nada.
		"""

		print("\n\n------- TEST 'Χ^2' -------\n")

		self.chi_squared = np.sum(((self.y-self.model_y)/self.y_err)**2)
		self.p_chi = stats.chi2.sf(self.chi_squared, len(self.x) - 1 - len(self.popt))
		print('chi^2: ' + str(self.chi_squared))
		print('p-valor del chi^2: ' + str(self.p_chi))

		if self.p_chi<0.05:
			print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
		else:
			print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')

	def test_f(self):
		"""
		Realiza el test 'F' de Snedecor/Fisher.
		Muestra en pantalla la información, no retorna nada.
		"""

		print("\n\n------- PRUEBA 'F'' DE FISHER -------\n")

		F = self.ESS*(len(self.x)-len(self.popt))/self.RSS/(len(self.popt)-1)
		p_f = stats.f.sf(F,len(self.popt)-1,len(self.x)-len(self.popt))
		
		print('Test F: ' + str(F))
		print('p-valor del F: ' + str(p_f))

		if p_f<0.05:
			print('Se rechaza la hipótesis de que el modelo ajuste tan bien como uno sin variables independientes.')
		else:
			print('No se puede rechazar la hipótesis de que el modelo ajuste tan bien como uno sin variables independientes.')

	def make_graph(self):
		"""
		Realiza el gráfico de manera rápida, con residuos y paleta de colores por defecto.
		Para mas personalización, se recomienda utilizar la clase Gráfico.
		"""
		
		#Creamos un objeto de clase labopy.Gráfico sin los parametros opcionales.
		Grafico(self.x,self.y,self.expr,self.y_err,res = True, hist = True, figsize = (16,4.5))

class Grafico(Ajuste):

	#Separar init en distintos métodos.
	def __init__(self, x: list,y: list, expr: sp.Basic, y_err: list, res: bool = False, hist: bool = False, figsize: tuple = (10,5), 
				 color_palette: list = ["#007b96", "#71a5b8", "#b9d1db", "#ffffff", "#c6c6c6", "#dcb5c9", "#b66e95", "#8c1f64"]):
		"""
		Clase que permite crear el ajuste de una manera mas personalizable. Como es hija de 'Ajuste', también permite realizar los tests de bondad a partir de esta.
		
		Parámetros:
		-	x: array-like -> Valores de la variable independiente.
		-	y: array-like -> Valores de la variable dependiente.
		-	y_err: array-like o int -> Cantidad de incerteza en la variable dependiente.
		-	expr: labopy.Expresion -> Expresión que almacena la forma funcional a ajustar.
		-	res: Bool (opcional, Default = False) -> Muestra el gráfico de residuos. Si este esta desactivado, el histograma también.
		-	hist: Bool (opcional, Default = False) -> Muestra el histograma para los residuos.
		-	figsize: tuple (opcional, Default = (10,5)) -> Tamaño del gráfico en pulgadas.
		-	color_palette: list (opcional, Default = ["#007b96", "#71a5b8", "#b9d1db", "#ffffff", "#c6c6c6", "#dcb5c9", "#b66e95", "#8c1f64"] ) -> Lista que guarda la paleta de 8 colores en hexagesimal para el gráfico.

		Notas: 
		1.	El primer símbolo que hayamos pasado al crear la expresión sera el utilizado como variable dependiente.
		2.	Se puede acceder siempre a los 'axes' o al 'fig' como 'self.axes' o 'self.fig' respectivamente para más personalización.
		3.  Al ser hijo de la clase labopy.Ajuste, podemos usar el metodo save_info para actualizar la información del gráfico.
		"""

		#Llamamos al _init_ de labopy.Ajuste.
		super().__init__(x,y,expr,y_err)

		#---------------------------------------------
		#Creación de la figura y el plot.

		if res == False: #No queremos residuos.

			self.fig = plt.figure(figsize = figsize, tight_layout = True)
			self.ax = self.fig.add_subplot(1,1,1)

		else: #Queremos residuos.

			if hist == True: #Queremos histograma.

				self.fig, self.axs = plt.subplots(1,3,tight_layout = True,figsize= figsize, gridspec_kw={'width_ratios': [2, 1, 0.5]})
				title_list = ["a)","b)","c)"]

			else: #No queremos histograma.

				self.fig, self.axs = plt.subplots(1,2,tight_layout = True,figsize= figsize, gridspec_kw={'width_ratios': [2, 1]})
				title_list = ["a)","b)"]

		#-------------------------------------------
		#Estilos.

		if res == False: #Si no tenemos residuos.

			self.ax.set_facecolor(color=color_palette[4])
			self.ax.grid(color=color_palette[3],linestyle="--")

		else: #Si tenemos residuos.

			for i, ax in enumerate(self.axs):
				ax.set_facecolor(color=color_palette[4])
				ax.set_title(title_list[i])

				if i != 2: #Graficos que no son el histograma.
					ax.grid(color=color_palette[3],linestyle="--")
				else: #Gráfico de histograma.
					ax.set_xticks([])
					ax.set_yticks([])

		#---------------------------------------
		#Ploteo de la info.

		smooth_x = np.linspace(self.x[0],self.x[-1],100) #Linspace suavizado con los valores de x.

		if res == True: #Si estamos mostrando los residuos.
			
			#Ajuste:
			self.axs[0].plot(smooth_x,self.test_function(smooth_x,*self.popt),"--",color=color_palette[7], label = "Ajuste")

			#Mediciones:
			self.axs[0].errorbar(self.x,self.y, yerr = self.y_err, fmt = "o", color="black" ,markerfacecolor = color_palette[0],
						 capsize = 3, capthick = 1, label= "Mediciones")

			#Residuos:
			self.axs[1].errorbar(self.x,self.y-self.test_function(self.x,*self.popt),fmt="o",label="Residuos",color="black",markerfacecolor=color_palette[7])
			
			#Media de los residuos:
			res_mean = np.mean(self.y - self.test_function(self.x,*self.popt))
			self.axs[1].axhline(res_mean, linestyle = "-", color = color_palette[2], lw = 2, label = "Media")

			if hist == True: #Si tenemos histograma.

				#Histograma:
				n, bins, _ = self.axs[2].hist(self.y - self.test_function(self.x,*self.popt), bins = "sturges", orientation = "horizontal",color = color_palette[5], rwidth = 0.95, edgecolor = "black")

				#Curva del histograma:
				bins_center = [(bins[i+1] + bins[i])/2 for i in range(len(bins)-1) ] #Centramos los valores de los bins.
				gaussian_popt, _ = curve_fit(gaussian,bins_center,n) #Ajustamos por una gaussiana.

				smooth_hist = np.linspace(bins[0],bins[-1],100) #Linspace suavizado con valores de los bins.
				self.axs[2].plot(gaussian(smooth_hist,*gaussian_popt),smooth_hist,"--",color=color_palette[1])
				self.axs[2].set_ylim(self.axs[1].get_ylim()) #Comparten 'ylim' para que no se rompa la escala.

			#Leyenda:
			self.axs[0].legend(loc="upper left",fontsize=13,fancybox=False,edgecolor="black",facecolor=color_palette[3])
			self.axs[1].legend(loc="upper left",fontsize=13,fancybox=False,edgecolor="black",facecolor=color_palette[3])

		else: #Si no queremos los residuos.

			#Ajuste:
			self.ax.plot(smooth_x,self.test_function(smooth_x,*self.popt),"--",color=color_palette[7], label = "Ajuste")

			#Mediciones:
			self.ax.errorbar(self.x,self.y, yerr = self.y_err, fmt = "o", color="black" ,markerfacecolor = color_palette[0],
						 capsize = 3, capthick = 1, label= "Mediciones")

			#Leyenda:
			self.ax.legend(loc="upper left",fontsize=13,fancybox=False,edgecolor="black",facecolor=color_palette[3])

class ExpresionError(Exception):

	def __init__(self):
		super().__init__()