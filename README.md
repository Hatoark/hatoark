# hatoark
3層ニューラルネットワーク(NN)の試作

## 使い方
### 各ノード数を決めてNNをインスタンス化
`NN3layers(入力ノード数, 中間ノード数, 出力ノード数);`

### 学習させる
`study(ArrayList<Integer> インプットのリスト(0or1), ArrayList<Integer> インプットに対する教師信号のリスト(0or1));`

入力データセットを複数回ループさせ学習させる

### 判定したいインプットを投入すると、結果を格納したリストが返ってくる
`getResult(ArrayList<Integer> 判定したい入力のリスト(0or1));`

出力層のリストが返ってくる。0.0～1.0の範囲の値が格納されている。
