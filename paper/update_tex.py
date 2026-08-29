import os

tex_content = r"""\documentclass[10pt,twocolumn,letterpaper]{article}

\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{subcaption}
\geometry{a4paper, margin=0.75in}

\title{\LARGE \bf
Topological Classification via Parametric Manifolds: An Evolutionary and Gradient-Descent Approach (FM, AM, GAM)
}

\author{Hugo Campos Castro}
\date{\today}

\begin{document}

\maketitle
\thispagestyle{empty}
\pagestyle{empty}

\begin{abstract}
Este artículo describe la evolución y diseño de una familia de clasificadores topológicos fundamentados en la optimización de variedades paramétricas en $\mathbb{R}^D$. El enfoque resuelve el problema de clasificación no lineal envolviendo distribuciones de datos en fronteras continuas. Presentamos la evolución cronológica del ecosistema: desde el modelo inicial trigonométrico \textbf{Fourier Manifold (FM)}, pasando por la optimización polinomial rápida del \textbf{Angular Manifold (AM)}, hasta la formulación completamente auditable del \textbf{Generalized Additive Manifold (GAM)}. Inicialmente, la topología de estas variedades fue optimizada estocásticamente mediante el motor genético multi-objetivo \textbf{NSGA-II} (optimizando Error vs. Complejidad de curva). Posteriormente, demostramos que las formulaciones AM y GAM poseen propiedades analíticas diferenciables, permitiendo un salto arquitectónico masivo hacia un \textbf{Descenso de Gradiente Multi-Objetivo (MGDA)}. Esta transición de NSGA-II hacia MGDA provee aceleraciones de convergencia de magnitudes superiores a $100\times$, permitiendo tiempos de entrenamiento en fracciones de segundo viables para Edge AI, reteniendo la exactitud topológica y proveyendo métricas transparentes de \textit{Feature Importance}.
\end{abstract}

\section{Introducción: El Viaje Topológico}
La separación de clases en espacios no linealmente separables es abordada clásicamente deformando el espacio (ej. truco del kernel en SVMs) o mediante millones de proyecciones distribuidas (Redes Neuronales). El presente trabajo aborda el problema de forma topológica inversa: construimos un colector paramétrico cerrado (\textit{manifold}) $M \subset \mathbb{R}^D$ que se pliega para encapsular a las clases en su espacio topológico nativo, permitiendo extraer explícitamente la ecuación matemática de la frontera (interpretabilidad analítica estricta o \textit{White-Box}).

La arquitectura ha atravesado cuatro fases de evolución funcional, cada una superando las vulnerabilidades de la anterior:
\begin{itemize}
    \item \textbf{Generación 1: Fourier Manifold (FM-NSGA-II).} La idea original utilizaba series de Fourier para tejer superficies de separación no-lineal. Al no ser analíticamente diferenciable respecto a un error discreto, se acopló al potente motor estocástico multi-objetivo NSGA-II \cite{deb2002fast}. Aunque logró alta expresividad, la pesada Unidad Aritmética Lógica (ALU) trigonométrica limitó su velocidad.
    \item \textbf{Generación 2: Angular Manifold (AM-NSGA-II).} Para mitigar el embotellamiento trigonométrico, se formuló una proyección direccional utilizando polinomios recursivos de Chebyshev. Esto aceleró drásticamente la inferencia, pero la evolución estocástica poblacional (NSGA-II) seguía siendo asintóticamente lenta para Big Data.
    \item \textbf{Generación 3: Generalized Additive Manifold (GAM-NSGA-II).} Se rediseñó el AM para independizar los ejes y dotar al algoritmo de auditabilidad nativa en altísima dimensionalidad, generando una ponderación explícita de \textit{Feature Importance} en hardware analítico médico o financiero.
    \item \textbf{Generación 4: El Salto Diferencial (MGDA).} El hallazgo crítico de que AM y GAM son diferenciables permitió descartar la selección natural de NSGA-II y adoptar el \textbf{Multiple Gradient Descent Algorithm (MGDA)}. Al utilizar gradientes exactos junto con optimizadores modernos (Adam, Focal Loss), el tiempo de convergencia colapsó de decenas de segundos a fracciones de segundo, cristalizando el método.
\end{itemize}

\section{Formulación Topológica (Los Kernels)}

\subsection{Generación 1: Fourier Manifold (FM)}
Definimos la frontera de decisión originaria como una hiper-superficie de Fourier. La aplicación continua $\mathbf{f}: [0, 2\pi) \to \mathbb{R}^D$ define la coordenada en la dimensión $d$ como:
\begin{equation}
    f_d(t) = a_{0,d} + \sum_{k=1}^{H} \left( a_{k,d} \cos(kt) + b_{k,d} \sin(kt) \right)
\end{equation}
donde $H \in \mathbb{N}$ controla la complejidad oscilatoria. La pertenencia topológica de un punto $\mathbf{x}$ se determina mediante la \textbf{distancia con signo radial} respecto al centroide de la clase $\mathbf{c}$:
\begin{equation}
    D_S(\mathbf{x}, M) = d_w(\mathbf{x},\, \mathbf{c}) - d_w(\mathbf{f}(t^*),\, \mathbf{c})
\end{equation}
donde $t^* = \arg\min_t d_w(\mathbf{x}, \mathbf{f}(t))$ y $d_w$ es la distancia euclidiana ponderada por los co-pesos del modelo $\mathbf{w}$.

\textbf{Ejemplo Didáctico (La Construcción de la Curva):} Pensemos en esto paso a paso. Si establecemos $H=0$, la ecuación elimina toda la sumatoria y nos queda simplemente $f_d(t) = a_{0,d}$, un punto. Si $H=1$, la ecuación introduce un seno y un coseno, trazando un \textbf{círculo o elipse perfecta} alrededor de los datos (la frontera de decisión más básica). Sin embargo, al permitir que el algoritmo evolutivo aumente a $H=3$, se introducen armónicos de alta frecuencia que literalmente "abollan" y "estiran" ese círculo original, permitiendo que la curva adquiera formas de herradura, medialuna o espiral para ajustarse herméticamente a un conjunto de datos complejo.

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/plot_chromosome.png}
    \caption{Representación del Cromosoma original de Fourier (FM). Contiene los pesos $\mathbf{w}$ para ponderación de \textit{features} y un arreglo de amplitud trigonométrica dinámico.}
    \label{fig:chrom_fm}
\end{figure}

\subsection{Generación 2: Angular Manifold (AM)}
Para combatir el sobrecosto de calcular $\sin$ y $\cos$ en cada punto, el genotipo AM proyecta una hipersuperficie desde un eje polar $\mathbf{v}$. El radio límite se define mediante polinomios de Chebyshev sobre la proyección direccional $z = \cos(\theta) = \frac{\langle \mathbf{x} - \mathbf{c}, \mathbf{v} \rangle_w}{\| \mathbf{x} - \mathbf{c} \|_w \| \mathbf{v} \|_w}$:
\begin{equation}
    R(\mathbf{x}) = a_0 + \sum_{k=1}^{H} a_k T_k(z)
\end{equation}
La distancia se simplifica a $D_S = \|\mathbf{x} - \mathbf{c}\|_w - R(\mathbf{x})$, una inferencia de orden $O(D+H)$ sin búsqueda de sub-parámetros $t$.

\textbf{Ejemplo Didáctico (La Compresión Dimensional):} Imagina que tenemos datos médicos en 30 dimensiones. En lugar de trazar una esfera compleja en 30D (muy costoso), el algoritmo AM descubre matemáticamente un único vector direccional "maestro" $\mathbf{v}$. Para evaluar a un nuevo paciente $\mathbf{x}$, el algoritmo simplemente "mira" qué ángulo forma el paciente con respecto a ese vector maestro. Luego, el polinomio de Chebyshev dicta la regla: \textit{"Si el paciente está a un ángulo de $45^\circ$, su radio límite máximo de pertenencia a la enfermedad es de 2.5 unidades"}. Hemos comprimido 30 dimensiones a una simple comprobación trigonométrica de un escalar.

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/plot_angular_chromosome.png}
    \caption{Representación del Cromosoma Angular (AM). Comprime la dimensionalidad a un vector direccional y utiliza coeficientes polinomiales para definir el radio límite.}
    \label{fig:chrom_am}
\end{figure}

\subsection{Generación 3: Gen. Additive Manifold (GAM)}
En vez de comprimir toda la dimensionalidad a un único escalar polar $z$, el GAM Manifold formula un polinomio aditivo independiente por cada característica $d$. La frontera radial se consolida como:
\begin{equation}
    R(\mathbf{x}) = a_0 + \sum_{d=1}^{D} w_d \left( \sum_{k=1}^{H} a_{d,k} T_k\!\left(\hat{x}_d\right) \right)
\end{equation}
donde $\hat{x}_d$ es la característica estandarizada. Esta aditividad aísla el comportamiento por eje, lo que permite extraer el vector $\mathbf{w}$ post-entrenamiento como un índice exacto de \textit{Feature Importance}.

\textbf{Ejemplo Didáctico (Aislamiento Aritmético):} Supongamos un problema 2D donde predecimos Obesidad usando Feature 1 (Perímetro Abdominal) y Feature 2 (Color de Ojos). El modelo entrena un polinomio para $d=1$ y otro para $d=2$. Durante el entrenamiento evolutivo o por gradiente, el motor descubre que el Color de Ojos ($d=2$) arroja ruido, por lo que castiga el multiplicador $w_2$ acercándolo a cero (ej. $w_2 = 0.001$). Al multiplicar el polinomio 2 por $0.001$, todo el término se anula en la sumatoria matemática. Así, al revisar el cromosoma final, vemos que $w_1 = 0.99$ y $w_2 = 0.001$, dándonos una explicación humana innegable de cómo el modelo toma sus decisiones.

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/plot_gam_chromosome.png}
    \caption{Representación del Cromosoma Aditivo (GAM). Extiende la representación a una matriz de coeficientes para aislar el comportamiento funcional por cada característica del espacio.}
    \label{fig:chrom_gam}
\end{figure}

\section{Metodología de Entrenamiento e Inferencia}

\subsection{Estrategia One-Vs-Rest (OvR)}
Para extrapolar la frontera intrínsecamente binaria de una curva cerrada hacia un clasificador Multi-Clase, empleamos la técnica \textbf{One-Vs-Rest (OvR)}. Por cada clase $K$ presente en el conjunto de entrenamiento, se entrena un colector paramétrico individual cuya única misión es envolver (etiquetar como $+1$) a la masa de datos de la clase $K$, rechazando absolutamente todas las demás clases (etiquetadas como $-1$).

\textbf{Ejemplo Didáctico:} En un problema de imágenes médicas con clases [Sano, Neumonía, COVID-19], el algoritmo instanciará independientemente tres fronteras topológicas distintas (tres modelos matemáticos $M_{Sano}$, $M_{Neumonia}$, $M_{COVID}$). El modelo $M_{COVID}$ tratará a los pacientes de Neumonía y Sanos como una única "Clase Enemiga".

\subsection{Inferencia Topológica (Distancia Radial con Signo)}
Al momento de clasificar puntos nuevos del conjunto de pruebas, cada punto $\mathbf{x}$ se somete a todos los $K$ colectores entrenados. La asignación de clase no recae en un umbral estricto dentro/fuera, sino que se emplea la \textbf{Distancia Radial con Signo} $D_S$:
\begin{equation}
    k^* = \arg\min_k \left( D_S(\mathbf{x}, M_k) \right)
\end{equation}

\textbf{Ejemplo Didáctico:} Retomando el ejemplo médico anterior, supongamos que entra el Paciente X al sistema. El evaluador calcula la distancia de este paciente a los bordes de cada modelo:
- $D_S$ en $M_{Sano} = +4.2$ (El paciente está muy "fuera" del radio de sanidad).
- $D_S$ en $M_{Neumonia} = -0.5$ (El paciente está "dentro" de la curva de neumonía, por lo tanto es negativo).
- $D_S$ en $M_{COVID} = -3.8$ (El paciente está "profundamente hundido" en el centro de la curva COVID).

Al aplicar $\arg\min_k$, el sistema selecciona $-3.8$. El paciente X es clasificado como COVID-19 con extrema certidumbre analítica.

\section{Motores de Optimización}

\subsection{Optimización Estocástica (NSGA-II)}
Inicialmente, la parametrización no convexa de los colectores obligó a utilizar un algoritmo evolutivo. Se forjó la optimización de Pareto minimizando dos objetivos conflictivos, $\mathbf{O} = [O_1, O_2]^T$:
\begin{itemize}
    \item \textbf{Objetivo 1 (Error Empírico):} Fracción de puntos mal encapsulados $O_1 = \frac{1}{N} \sum_i \mathbb{I} [ y_i \neq \text{sign}(-D_S) ]$.
    \item \textbf{Objetivo 2 (Complejidad - Ockham):} Minimiza la longitud de la curva y el número de armónicos $H$, con un término de regularización $L_1$ sobre $\mathbf{w}$ para forzar un barrido de características irrelevantes.
\end{itemize}

\subsection{Salto Diferencial (MGDA)}
En las generaciones AM y GAM, la ausencia de una integral no-resoluble habilitó la extracción analítica de los gradientes $\nabla O_1$ y $\nabla O_2$. Reemplazamos el NSGA-II con \textbf{MGDA}, donde la dirección de descenso conjunta óptima $\mathbf{d}$ es:
\begin{equation}
    \mathbf{d} = -\left( \alpha \nabla O_1 + (1-\alpha) \nabla O_2 \right)
\end{equation}

\textbf{Ejemplo Didáctico (El Conflicto de Gradientes):} Imaginemos que entrenamos la topología, y en la iteración 100, el vector gradiente del Error ($\nabla O_1$) le indica a los parámetros del polinomio: \textit{"Aumenten su valor hacia el Norte para abarcar un dato que se nos escapó"}. Sin embargo, el gradiente de Complejidad ($\nabla O_2$) reclama: \textit{"Si vamos al Norte, la curva se vuelve muy serpenteante e inestable; debemos ir al Sur"}. El MGDA computa el ángulo de conflicto entre ambos vectores y resuelve matemáticamente el punto de equilibrio óptimo $\mathbf{d}$ (ej. moverse hacia el Oeste), logrando que el modelo reduzca el error sin estirar excesivamente la topología.

\section{Evaluación Experimental Integral}

\input{tables.tex}

\subsection{Asimetría Computacional y Velocidad}
La Tabla \ref{tab:time} desvela el impacto gigantesco del salto evolutivo. Mientras que el \textbf{FM-NSGA-II} original en $Two Moons$ invertía miles de milisegundos penalizado por la trigonometría, la versión polinomial \textbf{AM-NSGA-II} lo redujo dramáticamente. 

Como se ilustra de manera global en la Figura \ref{fig:time} y Figura \ref{fig:benchmark}, el cambio de paradigma estocástico hacia \textbf{MGDA} (gradientes analíticos) destrozó los tiempos de forma transversal sin penalizar la exactitud. En un dataset médico denso como \textit{Breast Cancer (30D)}, la familia estocástica NSGA colapsa por la vastedad combinatoria. A su vez, \textbf{MGDA-GAM} lo resuelve virtualmente en fracciones de segundo. Esto ubica al \textbf{MGDA-GAM} matemáticamente a la par con clasificadores C++ puros (como SVM o Random Forest), con la distinción de que su frontera es completamente extraíble.

\begin{figure*}[t!]
    \centering
    \includegraphics[width=\linewidth]{../build/benchmark_acc.png}
    \caption{Benchmark Global (Accuracy). Se contrasta la familia Topológica propuesta frente a los modelos clásicos. MGDA conserva o mejora la tasa de exactitud descubierta originalmente por el costoso NSGA-II.}
    \label{fig:benchmark}
\end{figure*}

\begin{figure*}[t!]
    \centering
    \includegraphics[width=\linewidth]{../build/benchmark_time.png}
    \caption{Asimetría Computacional. La adopción del algoritmo MGDA para los kernels polinomiales (GAM y Angular) provee una aceleración colosal respecto a su homólogo biológico estocástico.}
    \label{fig:time}
\end{figure*}

\subsection{Análisis Visual de Variedades Topológicas}
Para comprender cómo el componente matemático se traduce en clasificación física, la Figura \ref{fig:dataset} muestra los colectores generados superpuestos en el espacio de características bidimensional. El kernel de Fourier (FM) teje envolventes herméticas alrededor del contorno natural de la distribución en forma de luna. 

La Navaja de Ockham paramétrica, definida por nuestro segundo objetivo ($O_2$), se manifiesta visualmente en la Figura \ref{fig:pareto}. Aquí observamos cómo el motor NSGA-II castiga los modelos que desarrollan "tentáculos" o bordes serpenteantes súper-entrenados, obligando a la población final de soluciones a balancearse en un equilibrio perfecto en el Frente de Pareto, desde modelos simples de alto error hasta modelos complejos y precisos.

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/plot_dataset.png}
    \caption{Proyección de los datasets 2D. El colector paramétrico clasifica distribuciones no-lineales determinando la pertenencia mediante su topología inherente.}
    \label{fig:dataset}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/plot_pareto_3kernels.png}
    \caption{Frentes estocásticos de Pareto en la convergencia terminal. El cruce entre el Error de Inferencia vs Complejidad Topológica.}
    \label{fig:pareto}
\end{figure}

\subsection{Transparencia Extrema: Feature Importance (GAM)}
Al aislar la influencia axial mediante la adición de polinomios independientes, el kernel GAM posee auditabilidad plena (White-Box). En la Figura \ref{fig:gam_imp}, extrajimos vectorialmente los pesos de dimensionalidad optimizados de \textit{Breast Cancer (30D)}. Tal y como explicamos en el \textit{Ejemplo Didáctico del GAM}, los ejes multiplicadores $w_d$ actúan como atenuadores de ruido. Podemos apreciar determinísticamente que el modelo suprimió más del 60\% del ruido diagnóstico y confió masivamente el resultado a factores morfológicos críticos sin requerir aproximaciones post-hoc opacas (como LIME o SHAP).

\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{../build/gam_importance.png}
    \caption{Feature Importance nativa y explicable extraída de los parámetros internos explícitos del GAM-Manifold.}
    \label{fig:gam_imp}
\end{figure}

\subsection{El Límite: La Maldición de Dimensionalidad}
Al someter a la familia de algoritmos al dataset sintético \textit{Adversarial Noise (52D)} —círculos bidimensionales ofuscados por 50 dimensiones de ruido extremo—, todos los modelos de distancias ponderadas (Topológicos y $k$-NN) se deprimieron hacia un $\sim$50\% Accuracy, simulando el azar. El espacio exploratorio para los pesos del NSGA-II o los gradientes en MGDA fue abrumado por el volumen euclidiano del hipercubo. Los únicos algoritmos capaces de extraer la señal real fueron los Árboles de Decisión, cuya bifurcación ortogonal actúa matemáticamente como un descarte topológico exacto para hiper-planos.

\section{Conclusión}
A lo largo de este documento, hemos desarrollado y comprobado una tesis clave: \textbf{la clasificación no lineal no requiere cajas negras indescifrables}. Como "apunte" final, los pilares centrales descubiertos en esta investigación son:
\begin{itemize}
    \item \textbf{Cajas Blancas Geométricas:} Reemplazar hiper-planos abstractos por superficies continuas (colectores) permite resolver topologías en herradura, luna y espiral sin perder la ecuación matemática gobernante.
    \item \textbf{La Trigonometría es Precisa pero Costosa:} El uso de Series de Fourier (FM) proporciona flexibilidad extrema, pero es ineficiente computacionalmente para grandes masas de datos ($O(N \log N)$ real).
    \item \textbf{El Aislamiento habilita la Auditoría:} Al transicionar de una trigonometría acoplada (FM) a polinomios independientes por eje (GAM), recuperamos un vector analítico puro capaz de indicar al experto humano exactamente \textit{qué variables} decantaron la clasificación médica o industrial.
    \item \textbf{El Salto Evolutivo a Diferencial:} Reemplazar el motor genético estocástico (NSGA-II) por un algoritmo de gradientes multi-objetivo (MGDA) generó aceleraciones superiores a $100\times$, consolidando esta tecnología paramétrica como una alternativa viable para inferencia embebida (Edge AI).
\end{itemize}

\begin{thebibliography}{99}
\bibitem{deb2002fast}
K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, ``A fast and elitist multiobjective genetic algorithm: NSGA-II,'' \textit{IEEE Transactions on Evolutionary Computation}, vol. 6, no. 2, pp. 182-197, 2002.
\end{thebibliography}
\end{document}
"""

with open("/home/uwo/Projects/RM-NSGA-II/paper/main.tex", "w") as f:
    f.write(tex_content)
