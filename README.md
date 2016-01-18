TimeSVD++ Implementation
=============

以 TimeSVD++ 實作 time-aware 電影推薦系統

## 檔案說明

* Main.scala: 主執行檔案
* package.scala: 定義 package object 以放置 global variable
* TrainingModel.scala: 所有功能的 abstract class
* MatrixFacotrization.scala: 純矩陣分解演算法
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
    * `3` : SVD++
    * `4` : timeSVD
    * `5` : timeSVD++
  * `steps` : Training iterations
  * `numFactors` : 矩陣分解的 factors 數量
  * `filePath` : Dataset檔案位置
  * `splitStr` : 每筆評分資料的分隔符號
3. 編譯 `scalac *.scala`
4. 執行 `scala -cp . Main`

## 程式說明

讀取電影評分檔案，取 70% 的 user 為 training data，其餘 30% 的 user 取其最近時間的評分作為 test data。

程式中實作了五種 SVD 系列的演算法：純矩陣分解演算法、SVD 演算法、SVD++ 演算法、沒有 implicit feedback 的 time-SVD演算法、與timeSVD++ 演算法。

### Input 檔案格式

電影評分scale

## TimeSVD++ 演算法

## 執行效能

Measure by root mean squared error (RMSE)

30 iterations

3692 ratings from 943 users on 1682 movies at 802 different dates

Number of test cases : 46

|Model|_f_=10|_f_=20|_f_=50|_f_=100|_f_=200|
|:---|:---:|:---:|:---:|:---:|:---:|
|Matrix Factorization|1.373|1.370|1.517|1.460|1.586|
|SVD|1.135|1.245|1.386|1.327|1.267|
|SVD++|1.150|1.269|1.320|1.459|1.458|
|timeSVD|1.672|1.544|1.693|1.481|1.725|
|timeSVD++||||||
