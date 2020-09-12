# Yomemi-Moemi-Classification-Sample
![GitHub](https://img.shields.io/github/license/nerrog/Yomemi-Moemi-Classification-Sample)
![Twitter Follow](https://img.shields.io/twitter/follow/nerrog_blog)

[ヨメミか萌実か判別するAI](app-yo-moe-ai.nerrog.net)のモデル生成に使用されたソースコードです

# 使用ライブラリ
[requirements.txt](requirements.txt)参照

*※GPU使わない場合は tensorflow-gpuを tensorflowに変えてインストールしてください*
# 開発環境
・Windows10 2004 64bit

・Anaconda 1.19.2 (Python3.8.5)

・AMD Ryzen5 3600

・RAM 32GB

・NVIDIA GeForce RTX 2060 SUPER
# 使用方法

## データセットの準備
まず`Moemi`、`Yomemi`フォルダにそれぞれ画像を集めてきます

次に

```
python3 img.py
```

を実行して画像ファイルをnumpyのバイナリに変換します

そうしたら`dataset.npy`が生成されます

## モデルのトレーニング

```
python3 train.py
``` 

これでモデルのトレーニングが開始されます

データセットの数や環境に応じてソース上の`epochs=20`の数値を変えてください

適切な値にしないと学習不足、過学習になる場合があります

トレーニングが終わると`cnn.h5`というkerasのモデルが生成されます

## モデルの使用方法

```
python3 think.py <画像ファイルへのパス>
```

引数に画像ファイルのパスを指定します

tensorflowが動き始めて確率と結果が表示されれば成功です