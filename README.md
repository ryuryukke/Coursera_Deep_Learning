# Coursera_Deep_Learning
This repository contains my own solutions to the programming assignments during Coursera [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) Course.

This is the solution to the latest problem as of summer 2020.

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

### Week 1 - [Initialization](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Improving%20Deep%20Neural%20Networks/week1/Initialization.ipynb)

パラメータ(W,b)の初期化の方法によって学習の進み方が異なることを観察します。パラメータの初期値がどれほどNNの学習速度に影響するのかを学びます。

#### ポイント
#### ・Wを全て0に初期化すると、、

複数のノードが同じ出力をしている(各層におけるノードの出力の分布(アクティベーション分布)が偏る)状態となる。→ わざわざ多くのノードを用いる必要がない → 隠れ層のユニット数が全て1のNNと同値となり、学習が進まない。

#### ・Wをランダムに初期化しても、その値が大きい場合

最後の層のsigmoidでの勾配が限りなく0に近くなり(勾配消失)、学習スピードが格段に遅くなってしまう。

#### ・He初期化

活性化関数にReLUを用いるときに適したパラメータの初期化手法。パラメータの初期化において、ノード数nに対して、平均0、標準偏差√(2/n)である正規分布から設定する。

そうすると、アクティベーション分布に偏りが生じにくくなり、学習がよく進む。

活性化関数をsigmoidやtanhを使う場合は、Xavierの初期化を行う。

### Week 1 - [Regularization](https://github.com/ryuryukke/Coursera_Deep_Learning/blob/master/Improving%20Deep%20Neural%20Networks/week1/Regularization_v2a.ipynb)

L2正規化とDropoutの2手法によってどれほど、過学習が軽減されるかを観察します。

#### ポイント
#### ・過学習を防ぐには？
1.学習データを増やす

2.ネットワークの容量を減らす

3.重みに対する正則化(L1, L2..)

4.Dropout

5.データ拡張(画像を反転したり..)

#### ・L2正規化

正規化項(各層ごとの行列Wの要素の二乗の和の総和)をコスト関数に付け加える。正規化項のλの値を大きくすれば、各層のパラメータWの値が小さくなる。

つまり、NNが単純化されロジスティック回帰に近づくことで、適度な線形表現が得られ、過学習(overfitting, high-variance)を回避できる。一方で、λの値を大きくしすぎると、過度に線形的になり、返って学習不足(underfitting, high-bias)な状態になりうる。

正則化項を加えたコスト関数から、dWを求めてWの更新の式を立てると、Wが任意の値で、Wが小さくなるように更新されることがわかる。これがL2正規化が重み減衰と呼ばれる所以である。

#### ・Dropout

学習の繰り返しのたび、決めた確率で、ある層の内のいくつかのノードを使わないようにして、学習を行っていく。

つまりは、毎iterationで、異なる小さなNNモデルで学習をしていることになる。複数の学習器を使わず、一つの学習器だけでアンサンブル学習が行えていることになる。
これが汎化性能向上につながる。

ノードの出力のスケールを合わせるために、訓練時にkeep_probで割る。(Inverted Dropout)

これは、Dropoutによって無効にしたノード分だけ、出力が小さくスケールされてしまうから。

### Week 1 - [Gradient Checking]()

自分が作ったbackpropがうまく動作しているかを知るために、各パラメータにおけるコスト関数の偏微分が正しい値かを知る必要がある。

この偏微分をbackpropではなく、実際に近似的に計算して、値が極めて近ければ、自分が作ったbackpropがうまく動作していることになる。

コンピュータでは、微分は行うことができない(コンピュータが扱える数は有限で、連続変数を取り扱うことが不可能である一方で、微分は連続変数のときにのみ意味のある操作である)ため数値微分という近似的な微分を行う。ここでは、中心差分による数値微分を行う。

### Week 2 - [Optimization]()(comming soon)
