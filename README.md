TimeSVD++ Implementation
=============

## 檔案說明

* Main.scala: 主執行檔案
* package.scala: 定義 package object 以放置 global variable
* TrainingModel.scala: 所有功能的 abstract class
* MatrixFacotrization.scala: 基本的矩陣分解
* SVD.scala: SVD 演算法
* SVDplus.scala: SVD++ 演算法
* TimeSVD.scala: 沒有 implicit feedback 的 time-SVD演算法
* TimeSVDplus.scala: timeSVD++ 演算法
* GenerateData.scala: 從原始Dataset隨機產生較小的Dataset

## Windows 環境執行

1. 將8個程式碼檔案 `Main.scala` `package.scala` `TrainingModel.scala` `MatrixFacotrization.scala` `SVD.scala` `SVDplus.scala` `TimeSVD.scala` `TimeSVDplus.scala` 置於同一資料夾中
2. 設定`package.scala`程式碼中變數
  * `selectAlgorithm` : 設定演算法
    * `1` : Matrix Factorization
    * `2` : SVD
  * `steps` : Training iterations
3. 編譯 `scalac *.scala`
4. 執行 `scala -cp . Main`

## TimeSVD++ 演算法

## 執行效能
RMSE
30 iterations
There are 3692 ratings from 943 users on 1682 movies
Number of test cases : 46
|Model|_f_=10|_f_=20|_f_=50|_f_=100|_f_=200|
|:---|:---:|:---:|:---:|:---:|:---:|
|Matrix Factorization|1.373|1.370|1.517|1.460|1.586|
|SVD|1.135|||||
|SVD++||||||
|timeSVD||||||
|timeSVD++||||||
