import scala.io.Source

package object function {

	case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

	//set algorithm, number of trainging iteration, 
	//and number of factors in matrix factorization
	val (selectAlgorithm, steps, numFactors) = (1, 30, 10)

	val (filePath, splitStr) = ("dataset/ratingsNetflix1.dat", "::") //從Netflix篩選出的100k檔案
	//val (filePath, splitStr) = ("dataset/ratingsNetflix2.dat", "::") //從Netflix篩選出的1m檔案
	//val (filePath, splitStr) = ("dataset/ratings20m.dat", "::") //從MovieLen 20m 篩選出的小檔案
	//val (filePath, splitStr) = ("../dataset/ml-100k/u.data", "\t") //MovieLen小檔案
	//val (filePath, splitStr) = ("../dataset/ml-1m/ratings.dat", "::") //MovieLen大檔案

	//讀取評分檔案，存為List，每一筆評分的資料結構存為RatingDataStructure
	val ratingFile = Source.fromFile(filePath)		
		.getLines
		.toList
		.map{line =>
			val fields = line.split(splitStr)
			val tempRating = RatingDataStructure(fields(0).toInt, fields(1).toInt, fields(2).toDouble, fields(3).toLong)
			tempRating
			}
	val numUsers = ratingFile 
	                .reduceLeft( (a,b) => if (a.userID > b.userID) a else b) 
	                .userID
	val numMovies = ratingFile 
	                 .reduceLeft( (a,b) => if (a.movieID > b.movieID) a else b) 
	                 .movieID				

	//Rating Matrix
	//!! how to initial
	val ratings = Array.fill(numUsers)(Array.fill(numMovies)(0.0))
	ratingFile.foreach{ x => ratings(x.userID - 1)(x.movieID - 1) =  x.rating }

}