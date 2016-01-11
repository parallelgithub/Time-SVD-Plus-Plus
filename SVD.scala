

import scala.io.Source
import scala.util.Random
import util.control.Breaks._
import scala.math

		case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

		val ratingFile = Source.fromFile("../GitHub/dataset/ml-100k/u.data")
			.getLines
			.toList
			.map{line =>
				//val fields = line.split("::")
				val fields = line.split("\t")
				val tempRating = RatingDataStructure(fields(0).toInt, fields(1).toInt, fields(2).toDouble, fields(3).toLong)
				tempRating
				}
		val numUsers = ratingFile 
		                .reduceLeft( (a,b) => if (a.userID > b.userID) a else b) 
		                .userID
		val numMovies = ratingFile 
		                 .reduceLeft( (a,b) => if (a.movieID > b.movieID) a else b) 
		                 .movieID				
		
		val ratings = Array.fill(numMovies)(Array.fill(numMovies)(0.0))
		ratingFile.foreach{ x => ratings(x.userID - 1)(x.movieID - 1) =  x.rating }

		val testStart = numUsers / 2
		val testUsers = new Array[Int](numUsers - testStart)
		val testMovies = new Array[Int](numUsers - testStart)
		val testRatings = new Array[Double](numUsers - testStart)
		for( index <- testStart until numUsers) {
			val recentMovieID = ratingFile
									.filter( _.userID - 1 == index ) 
									.reduceLeft( (a,b) => if (a.timestamp > b.timestamp) a else b) 
									.movieID 
			testUsers(index - testStart) = index 
			testMovies(index - testStart) = recentMovieID - 1
			testRatings(index - testStart) = ratings(index)(recentMovieID - 1)
			ratings(index)(recentMovieID - 1) = 0.0
		}


/*
val ratings = Array(Array(5,3,0,1),
	          Array(4,0,0,1),
	          Array(1,1,0,5),
	          Array(1,0,0,4),
	          Array(0,1,5,4)
	          )
*/

val n = ratings.length
val m = ratings(0).length
val k = 50


abstract class TrainingModel {

	protected val matrixP = Array.fill(n)(Array.fill[Double](k)(Random.nextDouble))
	protected val matrixQ = Array.fill(k)(Array.fill[Double](m)(Random.nextDouble))
	protected def dotProduct(userIndex: Int, movieIndex: Int) = {
		var sum = 0.0
		for(h <- 0 until k) {
			sum = sum + matrixP(userIndex)(h) * matrixQ(h)(movieIndex)
		}	
		sum		
	}

	def predict(userIndex: Int, movieIndex: Int): Double 
}

//http://blog.csdn.net/zhaoxinfan/article/details/8821419
//http://sifter.org/~simon/journal/20061211.html
//"Matrix factorization techniques for recommender systems", 2009 
class SVD extends TrainingModel {

	val steps = 5000
	//??
	//(0.001, 0.02) (0.01, 0.05) (0.015, 0.015)
	val (gamma, lambda) = (0.001, 0.02)

	val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	//!! how to init
	val userDeviation = Array.fill(n)(0.0)
	val movieDeviation = Array.fill(m)(0.0)

	def predict(userIndex: Int, movieIndex: Int) = { 
		overallAverage + userDeviation(userIndex) + movieDeviation(movieIndex) + dotProduct(userIndex, movieIndex)

	}

	private def gradientDescent(): Double = {

		

		for(u <- 0 until n ; i <- 0 until m){
			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)

				//!!
				val bu = userDeviation(u)
				userDeviation(u) += gamma * (eui * movieDeviation(i) - lambda * userDeviation(u))
				movieDeviation(i) += gamma * (eui * bu - lambda * movieDeviation(i))
				for(h <- 0 until k){
					val puh = matrixP(u)(h)
					matrixP(u)(h) += gamma * ( eui * matrixQ(h)(i) - lambda * matrixP(u)(h))
					matrixQ(h)(i) += gamma * ( eui * puh - lambda * matrixQ(h)(i))
				}
			}
		}
		
		var error = 0.0
		for(u <- 0 until n; i <- 0 until m){
			if (ratings(u)(i) > 0){ //??
				val tempDot = ratings(u)(i) - predict(u,i)
				error = error + tempDot * tempDot

				val bu2 = userDeviation(u) * userDeviation(u)
				val bi2 = movieDeviation(i) * movieDeviation(i)
				error = error + bu2 + bi2
				for(h <- 0 until k){					
					val pu2 = matrixP(u)(h)*matrixP(u)(h)
					val qi2 = matrixQ(h)(i)*matrixQ(h)(i)
					//!! parameter
					error = error + (lambda/2.0) * ( pu2 + qi2 )
				}
			}
		}
		error  
	}

	for(oneStep <- 1 to steps){				
		println("Training step " + oneStep)
		if( gradientDescent() < 0.001 )
			break
	}

}

// matrix factorization
// http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
class MatrixFacotrization extends TrainingModel {

	val steps = 5000
	val alpha = 0.0002
	val beta = 0.2		

	def predict(userIndex: Int, movieIndex: Int) = dotProduct(userIndex, movieIndex)

	private def gradientDescent(): Double = {
		for(i <- 0 until n ; j <- 0 until m)
			if (ratings(i)(j) > 0){
				val eij = ratings(i)(j) - dotProduct(i,j)
				for(h <- 0 until k){
					val pih = matrixP(i)(h)
					matrixP(i)(h) += alpha*(2 * eij * matrixQ(h)(j) - beta * matrixP(i)(h))
					matrixQ(h)(j) += alpha*(2 * eij * pih - beta * matrixQ(h)(j))
				}
			}
			
		
		var error = 0.0
		for(i <- 0 until n; j <- 0 until m)
			if (ratings(i)(j) > 0){
				val tempDot = ratings(i)(j) - dotProduct(i,j)
				error = error + tempDot * tempDot
				for(h <- 0 until k){
					error = error + (beta/2.0) * (matrixP(i)(h)*matrixP(i)(h)+matrixQ(h)(j)*matrixQ(h)(j))
				}
			}
		error  
	}

	for(oneStep <- 1 to steps){				
		if( gradientDescent() < 0.001 )
			break
	}

}


/*
val matrix = new MatrixFacotrization
for(i <- 0 until n ) {
	for(j <- 0 until m)
		print(matrix.predict(i,j) + " ")
	 println
}
*/


		var mae: Double = 0.0
		var maeCount: Int = 0

		//val matrix = new MatrixFacotrization
		val matrix = new SVD

		for( i <- testStart until numUsers) {
			//testUsers(i - testStart) = i 
			val movieIndex = testMovies(i - testStart)
			val actualRating = testRatings(i - testStart)
			
			val predictRating = matrix.predict(i, movieIndex)
			/*
				value match {
					case v if v < 1.0 => 1.0
					case v if v > 5.0 => 5.0
					case _ => value
				}			
				*/
			println(actualRating + " " + predictRating)
			mae = mae + math.abs(actualRating - predictRating)
			maeCount = maeCount + 1			
		}
		println("MAE = " + "%.3f".format(mae / maeCount) )

/*
MartrixFactorization
beta = 0.02
k = 2 : 0.757
k = 3 : 0.761
k = 4 : 0.755
k = 5 : 0.793

beta = 0.2
k = 2 : 0.759
k = 3 : 0.749
k = 4 : 0.758
k = 5 : 0.762

SVD
gamma = 0.0002
lambda = 0.02
k = 2 : 0.779
gamma, lambda = (0.01, 0.05)
k = 2 : 0.781
(gamma, lambda) = (0.015, 0.015)
k = 10 : 0.982
(gamma, lambda) = (0.001, 0.02)
k = 50 : 0.855

*/		