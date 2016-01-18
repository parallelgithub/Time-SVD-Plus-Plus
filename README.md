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

1. 將8個scala程式碼檔案 `Main.scala` `package.scala` `TrainingModel.scala` `MatrixFacotrization.scala` `SVD.scala` `SVDplus.scala` `TimeSVD.scala` `TimeSVDplus.scala` 置於同一資料夾中
2. 編譯 `scalac *.scala`
3. 執行 `scala -cp . Main`

## TimeSVD++ 演算法

## 執行效能