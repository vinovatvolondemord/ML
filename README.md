# Метод главных компонент

Еще одно решение проблемы мультиколлинеарности заключается в том,
чтобы подвергнуть исходные признаки некоторому функциональному преобразованию, гарантировав линейную независимость новых признаков, и,
возможно, сократив их количество, то есть уменьшив размерность задачи.


В **методе главных компонент** *(principal component analysis, PCA)* строится
минимальное число новых признаков, по которым исходные признаки восстанавливаются линейным преобразованием с минимальными погрешностями.
PCA относится к методам обучения без учителя (unsupervised learning), поскольку матрица «объекты–признаки» F преобразуется без учета целевого
вектора y.

Важно отметить, что PCA подходит и для регрессии, и для классификации,
и для многих других типов задач анализа данных, как вспомогательное преобразование, позволяющее определить эффективную размерность исходных
данных.

**Постановка задачи.** Рассмотрим матрицу «объектов–признаков»

![фото 1](https://github.com/serega14736/ML/blob/master/формула1.png)

Обозначим через <a href="https://www.codecogs.com/eqnedit.php?latex=z_i&space;=&space;(g_1(x_i)),\cdots&space;g_m(x_1))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_i&space;=&space;(g_1(x_i)),\cdots&space;g_m(x_1))" title="z_i = (g_1(x_i)),\cdots g_m(x_1))" /></a> признаковые описания тех же
объектов в новом пространстве <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{Z}&space;=&space;\mathbb{R}^m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{Z}&space;=&space;\mathbb{R}^m" title="\mathbb{Z} = \mathbb{R}^m" /></a> меньшей размерности <a href="https://www.codecogs.com/eqnedit.php?latex=m&space;<&space;n:" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m&space;<&space;n:" title="m < n:" /></a>

![фото 2](https://github.com/serega14736/ML/blob/master/формула2.png)

Потребуем, чтобы исходные признаковые описания можно было восстановить
по новым описаниям с помощью некоторого линейного преобразования,
определяемого матрицей <a href="https://www.codecogs.com/eqnedit.php?latex=U&space;=&space;(u_j_s)_n\times_m:" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U&space;=&space;(u_j_s)_n\times_m:" title="U = (u_j_s)_n\times_m:" /></a>

![фото 3](https://github.com/serega14736/ML/blob/master/формула3.png)

или в векторной форме: <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{x}&space;=&space;xU^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{x}&space;=&space;xU^T" title="\widehat{x} = xU^T" /></a>

Восстановленное описание <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{x}" title="\widehat{x}" /></a> xˆ не обязано в точности совпадать с исходным описанием <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>, но их отличие на объектах обучающей выборки должно быть
как можно меньше при выбранной размерности <a href="https://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m" title="m" /></a>.  Будем одновременно искать матрицы G (описание объектов в новом пространстве) и U (матрица
линейного преобразования) таким образом, чтобы суммарная невязка 

![фото 4](https://github.com/serega14736/ML/blob/master/формула4.png)

была минимальна, при условии, что все нормы евклидовы: <a href="https://www.codecogs.com/eqnedit.php?latex=||A||^2&space;=&space;tr&space;AA^T&space;=&space;trA^TA." target="_blank"><img src="https://latex.codecogs.com/gif.latex?||A||^2&space;=&space;tr&space;AA^T&space;=&space;trA^TA." title="||A||^2 = tr AA^T = trA^TA." /></a>

Пусть матрицы G и U невырождены и <a href="https://www.codecogs.com/eqnedit.php?latex=rk&space;G&space;=&space;rk&space;U&space;=&space;m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?rk&space;G&space;=&space;rk&space;U&space;=&space;m" title="rk G = rk U = m" /></a>

**Теорема**
Если <a href="https://www.codecogs.com/eqnedit.php?latex=m&space;\leq&space;rk&space;F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m&space;\leq&space;rk&space;F" title="m \leq rk F" /></a> то минимум <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta^2(G,U)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta^2(G,U)" title="\Delta^2(G,U)" /></a> достигается, когда столбцы матрицы U есть собственные векторы <a href="https://www.codecogs.com/eqnedit.php?latex=F^TF" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^TF" title="F^TF" /></a>, , соответствующие m максимальным
собственным значениям. При этом G = F U, матрицы U и G ортогональны.

Собственные векторы <a href="https://www.codecogs.com/eqnedit.php?latex=u_1,\cdots,u_m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_1,\cdots,u_m" title="u_1,\cdots,u_m" /></a> , отвечающие максимальным собственным
значениям, называют **главными компонентами**.

Рассмотрим некоторые свойства метода главных компонент.

Если <a href="https://www.codecogs.com/eqnedit.php?latex=m=n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m=n" title="m=n" /></a>, то <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta^2(G,U)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta^2(G,U)=0" title="\Delta^2(G,U)=0" /></a>. В этом случае представление <a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;GU^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;GU^T" title="F = GU^T" /></a> является точным и совпадает с сингулярным разложением: 

<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;GU^T&space;=&space;VDU^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;GU^T&space;=&space;VDU^T" title="F = GU^T = VDU^T" /></a>

если положить <a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;VD" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;VD" title="G = VD" /></a> и <a href="https://www.codecogs.com/eqnedit.php?latex=\Lambda&space;=&space;D^2&space;=&space;diag(\lambda_1,\cdots,\lambda_m)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Lambda&space;=&space;D^2&space;=&space;diag(\lambda_1,\cdots,\lambda_m)." title="\Lambda = D^2 = diag(\lambda_1,\cdots,\lambda_m)." /></a>
