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

每行為一筆評分資料，格式如下

  user ID | 電影 id | 評分 | timestamp

分隔符號可為任意字串，但須需指定至程式碼中 parse

電影評分範圍為 1.0 stars - 5.0 stars

## TimeSVD++ 演算法

每個SVD系列的演算法中需設計三種公式

1. 訂定 prediction rule
2. 根據前一步驟訂定 regularized squared error function
3. 將前一步驟的 function 最小化即可達到所求，因此我們使用 gradient descent scheme 求局部最小值。計算偏微分後得到每個變數的更新公式

矩陣中每個項目為所求的變數，初始值設為隨機亂數，
執行 training 時程式反覆計算每個變數的更新公式，
最後得到的結果即為 prediction rule model。

### TimeSVD++ prediction rule

純矩陣分解演算法的 prediction rule 只求相乘的兩個矩陣 p, q

> predict(u,i) = p_u * q_i

SVD演算法的 prediction rule 在純矩陣分解上加了 baseline estimates。

> predict(u,i) = mui + b_u + b_i + p_u * q_i
mui為所有評分的平均，b_u為 user u 平均評分與所有評分的偏差，b_i為電影 i 平均評分與所有評分的偏差

SVD++演算法在 SVD 上再加上 implicit feedback : y_i
> predict(u,i) = mui + b_u + b_i + [p_u + sum_y_i / sqrt(N(u))] * q_i
implicit feedback 的形式可以是購買紀錄、瀏覽紀錄、搜尋行為模式、滑鼠行為模式。在缺乏此類資訊的電影評分資料中，我們取 user 是否有對電影評分的 boolean 資訊(有評分即設1，反之設0)。

TimeSVD++演算法用時間對 SVD 中的某些變數參數化，以表現出變數在不同時間時的樣貌
>  predict(u,i)(t) = mui + b(t)_u + b(t)_i + [p(t)_u + sum_y_i / sqrt(N(u))] * q_i

>  b(t)_i = b_i + b_i,Bin(t)

>  b(t)_u = b_u + alpha_u * dev(t)_u + b_u,t
>  dev(t)_u = sign(t - t_u) * | t - t_u |bata

  p(t)_u = p_u + alpha_u * dev(t)_u + p_u,t

## 執行效能

Measure by root mean squared error (RMSE)

30 iterations

3692 ratings from 943 users on 1682 movies at 802 different dates

Number of test cases : 46

|Model|_f_=10|_f_=20|_f_=50|_f_=100|_f_=200|
|:---|:---:|:---:|:---:|:---:|:---:|
|Matrix Factorization|1.373|1.370|1.517|1.460|1.586|
|SVD|1.135|1.245|1.386|1.327|1.267|
|SVD++|1.127|1.034|1.184|1.464|1.617|
|timeSVD|1.672|1.544|1.693|1.481|1.725|
|timeSVD++|1.454|1.840|1.899|1.985|1.558|
