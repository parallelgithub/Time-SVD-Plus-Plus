TimeSVD++ Implementation
=============

實作 Time-aware 電影推薦系統，以 TimeSVD++ 演算法捕捉 user 在不同時間的喜好變化。

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

> predict(u,i)(t) = mui + b(t)_u + b(t)_i + [p(t)_u + sum_y_i / sqrt(N(u))] * q_i

每個參數化的變數描述如下：

* 原本的 b_i 分開為穩定的部分與隨時間變化的部分，由於電影平均偏差變化不大，因此每隔一段時間才記錄偏差變化

 > b(t)_i = b_i + b_i,Bin(t)

* b_u 分為穩定的部分與另外兩個隨時間變化的部分。dev(t)_u 記錄時間平緩的變化(線性變化)，越接近當前、值越大；b_u,t 記錄因時間不同產生的急遽變化部分，比如 user 不同天的評分會受當天心情影響，因此這部份的時間參數以一天為單位。

 > b(t)_u = b_u + alpha_u * dev(t)_u + b_u,t
 
 > dev(t)_u = sign(t - t_u) * abs(t - t_u)^bata

* p_u (user 的偏好) 也用與 b_u 類似的作法參數化

 > p(t)_u = p_u + alpha_u * dev(t)_u + p_u,t

* q_i (電影的特性) 不隨時間變化

## 程式說明

讀取電影評分檔案，取 70% 的 user 為 training data，其餘 30% 的 user 取其所有評分資料中最近時間的評分作為 test data。

程式中實作了五種 SVD 系列的演算法：純矩陣分解演算法、SVD 演算法、SVD++ 演算法、沒有 implicit feedback 的 time-SVD演算法、與timeSVD++ 演算法。

變數中的時間維度可用程式語言中內建的資料結構 Map 來處理。本專案使用了 scala 的 HashMap。

## 執行效能

Measure by root mean squared error (RMSE)

30 iterations

### 100K MovieLens Dataset

100000 ratings from 943 users on 1682 movies on 1 day

Number of test users : 282

|Model|_f_=10|_f_=20|_f_=50|_f_=100|_f_=200|
|:---|:---:|:---:|:---:|:---:|:---:|
|Matrix Factorization|1.103|1.062|1.316|1.613|1.692|
|SVD|1.071|1.097|1.210|1.319|1.434|
|SVD++|1.086|1.113|1.177|N/A|N/A|
|timeSVD|1.346|1.447|1.454|N/A|N/A|
|timeSVD++|1.195|1.279|1.394|N/A|N/A|

### 100K Dataset from Netflix

3692 ratings from 943 users on 1682 movies on **802 different days**

Number of test users : 46

|Model|_f_ = 10|_f_ = 20|_f_ = 50|_f_ = 100|_f_ = 200|
|:---|:---:|:---:|:---:|:---:|:---:|
|Matrix Factorization|1.373|1.370|1.517|1.460|1.586|
|SVD|1.135|1.245|1.386|1.327|1.267|
|SVD++|1.127|1.034|1.184|1.464|1.617|
|timeSVD|1.672|1.544|1.693|1.481|1.725|
|timeSVD++|1.633|1.621|1.991|1.438|1.763|

### 1M Dataset from Netflix

55450 ratings from 6039 users on 3900 movies on 1679 days

Number of test users : 302

|Model|_f_ = 200|
|:---|:---:|
|timeSVD++|1.802|