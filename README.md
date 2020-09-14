# Coursera_Deep_Learning
This repository contains my own solutions to the programming assignments during Coursera [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) Course.

## Course 1 - Neural Networks and Deep Learning

### Week 2 - [Python basics with numpy (Optional)](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Neural%20Network%20and%20Deep%20Learning/Week2/Python_Basics_With_Numpy_v3a.ipynb)

sigmoid関数やsoftmax関数、画像のベクトル化などを例に挙げながら、numpyの基礎を学びます。

### Week 2 - [Logistic Regression with a Neural Network mindset](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Neural%20Network%20and%20Deep%20Learning/Week2/Logistic_Regression_with_a_Neural_Network_mindset_v6a.ipynb)

ロジスティック回帰モデルを構築して、猫の画像判定を行います。

### Week 3 - [Planar data classification with a hidden layer](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Neural%20Network%20and%20Deep%20Learning/Week3/Planar_data_classification_with_onehidden_layer_v6c.ipynb)

一つの隠れ層を持つニューラルネットのモデルを構築して、2クラス分類を行います。

### Week 4 - [Building your deep neural network: Step by Step](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Neural%20Network%20and%20Deep%20Learning/Week4/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb)

多層のニューラルネットのモデル構築をします。

### Week 4 - [Deep Neural Network Application](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Neural%20Network%20and%20Deep%20Learning/Week4/Deep%2BNeural%2BNetwork%2B-%2BApplication%2Bv8.ipynb)

作った多層ニューラルネットを用いて、再び猫の画像判定を行い、判定精度の向上を観察します。

## Course 2 - Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

### Week 1 - [Initialization]()

パラメータ(W,b)の初期化の方法によって学習の進み方が異なることを観察します。パラメータの初期値がどれほどNNの学習速度に影響するのかを学びます。

#### キー
#### ・Wを全て0に初期化すると、、

複数のノードが同じ出力をしている(各層におけるノードの出力の分布(アクティベーション分布)が偏る)状態となる。→ わざわざ多くのノードを用いる必要がない → 隠れ層のユニット数が全て1のNNと同値となり、学習が進まない。

#### ・Wをランダムに初期化しても、その値が大きい場合

最後の層のsigmoidでの勾配が限りなく0に近くなり(勾配消失)、学習スピードが格段に遅くなってしまう。

#### ・He初期化

活性化関数にReLUを用いるときに適したパラメータの初期化手法。パラメータの初期化において、ノード数nに対して、平均0、標準偏差√(2/n)である正規分布から設定する。

そうすると、アクティベーション分布に偏りが生じにくくなる。

Week 1 - [Regularization]()

Week 1 - [Gradient Checking]()

