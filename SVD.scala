

import scala.io.Source
import scala.util.Random
//import util.control.Breaks._
import scala.math

		case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

		val ratingFile = Source.fromFile("../dataset/ml-100k/u.data")
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
		val numFactors = 2

		val ratings = Array.fill(numMovies)(Array.fill(numMovies)(0.0))
		ratingFile.foreach{ x => ratings(x.userID - 1)(x.movieID - 1) =  x.rating }

		//隨機從numUsers個users中挑出testSize個test users
		val testSize = 200
		val userCandidate = List.range(1, numUsers+1)
		def generateTestUsers(candidate: List[Int],count: Int, n: Int): List[Int] = {
			if(count == 0)
				Nil
			else{
				val i = Random.nextInt(n)	
				candidate(i) :: generateTestUsers((candidate.take(i) ::: candidate.drop(i+1)), count-1, n-1)
			}
		}
		//儲存 test user 的資料，並將預計測試的電影在matrix重設為0
		val testData = generateTestUsers(userCandidate, testSize, numUsers).map{ id =>
			val recent = ratingFile 
			              .filter( _.userID == id ) 
			              .reduceLeft( (a,b) => if (a.timestamp > b.timestamp) a else b)
					
			val actualRating = ratings(id - 1)(recent.movieID - 1)
			ratings(id - 1)(recent.movieID - 1) = 0.0
			RatingDataStructure(id, recent.movieID, actualRating, recent.timestamp )
		}


/*
val ratings = Array(Array(5,3,0,1),
	          Array(4,0,0,1),
	          Array(1,1,0,5),
	          Array(1,0,0,4),
	          Array(0,1,5,4)
	          )

val n = ratings.length
val m = ratings(0).length
val f = 3
*/

abstract class TrainingModel {

	protected val matrixP = Array.fill(numUsers)(Array.fill[Double](numFactors)(Random.nextDouble))
	protected val matrixQ = Array.fill(numFactors)(Array.fill[Double](numMovies)(Random.nextDouble))
	protected def dotProduct(userIndex: Int, movieIndex: Int) = {
		var sum = 0.0
		for(h <- 0 until numFactors) {
			sum = sum + matrixP(userIndex)(h) * matrixQ(h)(movieIndex)
		}	
		sum		
	}

	def predict(userIndex: Int, movieIndex: Int): Double 
}


// matrix factorization
// http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
class MatrixFacotrization extends TrainingModel {

	val steps = 5000
	val alpha = 0.0002
	val beta = 0.02		

	def predict(userIndex: Int, movieIndex: Int) = dotProduct(userIndex, movieIndex)

	private def gradientDescent(): Double = {
		for(i <- 0 until numUsers ; j <- 0 until numMovies)
			if (ratings(i)(j) > 0){
				val eij = ratings(i)(j) - dotProduct(i,j)
				for(h <- 0 until numFactors){
					val pih = matrixP(i)(h)
					matrixP(i)(h) += alpha*(2 * eij * matrixQ(h)(j) - beta * matrixP(i)(h))
					matrixQ(h)(j) += alpha*(2 * eij * pih - beta * matrixQ(h)(j))
				}
			}
			
		
		var error = 0.0
		for(i <- 0 until numUsers; j <- 0 until numMovies)
			if (ratings(i)(j) > 0){
				val tempDot = ratings(i)(j) - dotProduct(i,j)
				error = error + tempDot * tempDot
				for(h <- 0 until numFactors){
					error = error + (beta/2.0) * (matrixP(i)(h)*matrixP(i)(h)+matrixQ(h)(j)*matrixQ(h)(j))
				}
			}
		//println("Error: " + error)
		error  
	}

	for(oneStep <- 1 to steps){				
		gradientDescent()
			
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

//http://blog.csdn.net/zhaoxinfan/article/details/8821419
//http://sifter.org/~simon/journal/20061211.html
//"Matrix factorization techniques for recommender systems", 2009 
class SVD extends TrainingModel {

	val steps = 50000
	//??
	//(0.001, 0.02) (0.01, 0.05) (0.015, 0.015)
	val (gamma, lambda) = (0.002, 0.02)

	val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	//!! how to init
	val userDeviation = Array.fill(numUsers)(0.0)
	val movieDeviation = Array.fill(numMovies)(0.0)

	def predict(userIndex: Int, movieIndex: Int) = { 
		overallAverage + userDeviation(userIndex) + movieDeviation(movieIndex) + dotProduct(userIndex, movieIndex)

	}

	private def gradientDescent(): Double = {

		for(u <- 0 until numUsers ; i <- 0 until numMovies){
			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)

				userDeviation(u) += gamma * (eui - lambda * userDeviation(u))
				movieDeviation(i) += gamma * (eui - lambda * movieDeviation(i))
				for(h <- 0 until numFactors){
					val puh = matrixP(u)(h)
					matrixP(u)(h) += gamma * ( eui * matrixQ(h)(i) - lambda * matrixP(u)(h))
					matrixQ(h)(i) += gamma * ( eui * puh - lambda * matrixQ(h)(i))
				}
			}
		}
		
		var error = 0.0
		for(u <- 0 until numUsers; i <- 0 until numMovies){
			if (ratings(u)(i) > 0){ //??
				val tempDot = ratings(u)(i) - predict(u,i)
				error = error + tempDot * tempDot

				val bu2 = userDeviation(u) * userDeviation(u)
				val bi2 = movieDeviation(i) * movieDeviation(i)
				error = error + bu2 + bi2
				for(h <- 0 until numFactors){					
					val pu2 = matrixP(u)(h)*matrixP(u)(h)
					val qi2 = matrixQ(h)(i)*matrixQ(h)(i)
					
					error = error + (lambda/2.0) * ( pu2 + qi2 )
				}
			}
		}
		error  
	}

	for(i <- 1 to 100) gradientDescent()

	var last = math.abs(gradientDescent())
	var loop = true
	var i = 1
	while(loop){
		val err = math.abs(gradientDescent())
		println("Training step " + i + " : error = " + err)
		if(err > last)
			loop = false
		last = err
		i += 1
	}
/*
	var min = last
	var step = 0
	for(i <- 1 to steps){				
		val err = math.abs(gradientDescent())
		println("Training step " + i + " : error = " + err)
		if(err < min){
			min = err
			step = i
		}
	}
	println("min error : " + min + " at step " + step)
*/
}

//https://github.com/guoguibing/librec/blob/master/librec/src/main/java/librec/rating/SVDPlusPlus.java
//"Factorization meets the neighborhood- a multifaceted collaborative filtering model", 2008
class SVDPlus extends TrainingModel {

	val w = math.sqrt(numMovies)
	val steps = 5000
	//??
	val (gamma, lambda) = (0.002, 0.02)

	val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	//!! how to init
	val userDeviation = Array.fill(numUsers)(0.0)
	val movieDeviation = Array.fill(numMovies)(0.0)
	val Y = Array.fill(numFactors)(Array.fill[Double](numMovies)(Random.nextGaussian() * 0.1))

	def predict(userIndex: Int, movieIndex: Int) = { 

		var value = 0.0
		for(i <- 0 until numMovies)
			for(f <- 0 until numFactors)
				value += Y(f)(i) + matrixQ(f)(movieIndex)
		overallAverage + userDeviation(userIndex) + movieDeviation(movieIndex) + dotProduct(userIndex, movieIndex) + value / w
	}

	private def gradientDescent(): Double = {

		var error = 0.0

		for(u <- 0 until numUsers ; i <- 0 until numMovies){

			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)
				error += eui * eui

				userDeviation(u) += gamma * (eui - lambda * userDeviation(u))
				movieDeviation(i) += gamma * (eui - lambda * movieDeviation(i))
				error = error + userDeviation(u) * userDeviation(u) + movieDeviation(i) * movieDeviation(i)

				val sumYj = Y.map{x => x.reduceLeft(_ + _) / w}
				for(f <- 0 until numFactors){
					val puh = matrixP(u)(f)
					matrixP(u)(f) += gamma * ( eui * matrixQ(f)(i) - lambda * matrixP(u)(f))
					matrixQ(f)(i) += gamma * ( eui * (puh + sumYj(f)) - lambda * matrixQ(f)(i))
					//!! parameter
					error += (lambda/2.0) * ( matrixP(u)(f)*matrixP(u)(f) + matrixQ(f)(i)*matrixQ(f)(i) )
					for(j <- 0 until numMovies){
						val yj = Y(f)(j)
						Y(f)(j) += gamma * ( eui * matrixQ(f)(i) / w - lambda * yj)
						error += (lambda/2.0) * yj * yj
					}
				}
			}
		}

		error  
	}

	var min = math.abs(gradientDescent())
	var step = 0
	for(i <- 1 to steps){				
		val err = math.abs(gradientDescent())
		println("Training step " + i + " : error = " + err)
		if(err < min){
			min = err
			step = i
		}
	}
	println("min error : " + min + " at step " + step)

}

		var mae: Double = 0.0
		var maeCount: Int = 0

		val select = 2
		
		//Training
		val matrix = select match {
			case 1 => new MatrixFacotrization
			case 2 => new SVDPlus
		}

		//def test(model: TrainingModel): Double { -1.0 }

		//Test
		for(test <- testData){

			val actualRating = test.rating
			
			val predictRating = matrix.predict(test.userID-1, test.movieID - 1)
			/*
				value match {
					case v if v < 1.0 => 1.0
					case v if v > 5.0 => 5.0
					case _ => value
				}			
				*/

			println("User " + test.userID + " with movie " + test.movieID + " : ")
			println(" Predic rating " + "%.3f".format(predictRating) )
			println(" Actual rating " + test.rating)
			println				

			mae = mae + math.abs(actualRating - predictRating)
			maeCount = maeCount + 1			
		}
		println("MAE = " + "%.3f".format(mae / maeCount) )

/*
Step = 5000
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
(gamma, lambda) = (0.0002, 0.02)
f = 2 : 0.667

SVD step=until local minimal
(gamma, lambda) = (0.002, 0.02)
f = 20
testSize = 200
 MAE : 0.863 by 66892 steps in 7257.7 seconds 

f = 2
 MAE : 0.744 by 572 steps
*/		