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

