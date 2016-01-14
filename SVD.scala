

import scala.io.Source
import scala.util.Random
import scala.math
import java.util.concurrent.TimeUnit

		case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

		//val ratingFile = Source.fromFile("../dataset/ml-100k/u.data")
		val ratingFile = Source.fromFile("../dataset/ml-1m/ratings.dat")
			.getLines
			.toList
			.map{line =>
				//val fields = line.split("\t")
				val fields = line.split("::")
				val tempRating = RatingDataStructure(fields(0).toInt, fields(1).toInt, fields(2).toDouble, fields(3).toLong)
				tempRating
				}
		val numUsers = ratingFile 
		                .reduceLeft( (a,b) => if (a.userID > b.userID) a else b) 
		                .userID
		val numMovies = ratingFile 
		                 .reduceLeft( (a,b) => if (a.movieID > b.movieID) a else b) 
		                 .movieID				
		//number of factors in matrix factorization
		val numFactors = 2

		val ratings = Array.fill(numUsers)(Array.fill(numMovies)(0.0))
		ratingFile.foreach{ x => ratings(x.userID - 1)(x.movieID - 1) =  x.rating }

		//隨機從numUsers個users中挑出testSize個test users
		val testSize = (0.3 * numUsers).toInt
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

//
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

	val steps = 500
	val alpha = 0.0002
	val beta = 0.02		
	//http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

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

	val steps = 500

	val (gamma, lambda) = (0.002, 0.02)
	//(0.001, 0.02) 
	//(0.01, 0.05) http://blog.csdn.net/zhaoxinfan/article/details/8821419
	//(0.015, 0.015) 2008 (SVD) A Guide to Singular Value Decomposition for Collaborative Filtering

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

				//!! lambda
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
*/
}

//https://github.com/guoguibing/librec/blob/master/librec/src/main/java/librec/rating/SVDPlusPlus.java
//https://www.quora.com/Whats-the-difference-between-SVD-and-SVD++
//"Factorization meets the neighborhood- a multifaceted collaborative filtering model", 2008
class SVDPlus extends TrainingModel {

	val nFB = numMovies 
	def w(value: Double): Double = value * 1000.0

	val steps = 100
	val (gamma, lambda1, lambda2) = (0.007, 0.005, 0.015)
	//(0.007, 0.005, 0.015) "Factorization meets the neighborhood- a multifaceted collaborative filtering model"

	val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	//!! how to init
	val userDeviation = Array.fill(numUsers)(0.0)
	val movieDeviation = Array.fill(numMovies)(0.0)
	//According LibRec, initial by Gaussian distribution with mean 0.0 and standard deviation 1.0
	val feedback = Array.fill(numFactors)(Array.fill[Double](nFB)(Random.nextGaussian() * 0.1))
	val ratedMovieOfUsers = ratings.map{ x => 
		                        val s = x.size 
		                        for(i <- 0 until s if x(i) > 0.0) 
		                        	yield i 
		                        }
	//ratedMovieOfUsers.foreach{ x=> println(x) }

	def predict(userIndex: Int, movieIndex: Int) = { 

		var value = 0.0

		for(j <- ratedMovieOfUsers(userIndex))
			for(f <- 0 until numFactors)
				value += feedback(f)(j) + matrixQ(f)(movieIndex)

		//mui + bu + bi + qi*[pu + sum(yj)]		
		overallAverage + 
		  userDeviation(userIndex) + movieDeviation(movieIndex) + 
		  dotProduct(userIndex, movieIndex) + value / w(ratedMovieOfUsers(userIndex).size)
	}

	private def gradientDescent(): Double = {

		//update each factor
		for(u <- 0 until numUsers ; i <- 0 until numMovies){

			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)

				//update bu and bi
				userDeviation(u) += gamma * (eui - lambda1 * userDeviation(u))
				movieDeviation(i) += gamma * (eui - lambda1 * movieDeviation(i))

				val sumFBj = feedback.map{x => x.reduceLeft(_ + _) / w(ratedMovieOfUsers(u).size)}
				
				for(f <- 0 until numFactors){
					val puf = matrixP(u)(f)
					//update pu and qi
					matrixP(u)(f) += gamma * ( eui * matrixQ(f)(i) - lambda2 * matrixP(u)(f))
					matrixQ(f)(i) += gamma * ( eui * (puf + sumFBj(f)) - lambda2 * matrixQ(f)(i))
					
					//update yj					
					for(j <- ratedMovieOfUsers(u))
						feedback(f)(j) += gamma * ( eui * matrixQ(f)(i) / w(ratedMovieOfUsers(u).size) - lambda2 * feedback(f)(j))
					
				}
			}
		}

		var error = 0.0

		//comput error
		for(u <- 0 until numUsers ; i <- 0 until numMovies){

			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)
				val bu = userDeviation(u)
				val bi = movieDeviation(i)
				error += eui * eui + (lambda1/2.0) * ( bu * bu + bi * bi )

				for(f <- 0 until numFactors){
					//!! parameter
					error += (lambda2/2.0) * ( matrixP(u)(f)*matrixP(u)(f) + matrixQ(f)(i)*matrixQ(f)(i) )
					for(j <- ratedMovieOfUsers(u))	
						error += (lambda2/2.0) * feedback(f)(j) * feedback(f)(j)
					
				}
			}
		} //end of for(u;i)

		error  
	} //end of def gradientDescent()

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

/*
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
*/	
}

class TimeSVD extends TrainingModel {

	val steps = 500

	val (beta) = (0.4)
	val (gamma, lambda) = (0.002, 0.02)

	//!! how to init
	val userDeviation = Array.fill(numUsers)(0.0)
	val movieDeviation = Array.fill(numMovies)(0.0)
	val ratedMovieOfUsers = ratings.map{ x => 
		                        val s = x.size 
		                        for(i <- 0 until s if x(i) > 0.0) 
		                        	yield i 
		                        }

	val minStamp = ratingFile.reduceLeft( (a,b) => if (a.timestamp < b.timestamp) a else b).timestamp
	val maxStamp = ratingFile.reduceLeft( (a,b) => if (a.timestamp > b.timestamp) a else b).timestamp
	val numDays = days(maxStamp, minStamp) + 1
	val times = Array.fill(numUsers)(Array.fill[Long](numMovies)(0))
	//For the rating(u)(i) which we want to predict, its timestamp is preserved
	ratingFile.foreach{ x => times(x.userID - 1)(x.movieID - 1) =  x.timestamp }
	val userMeanDate = Array.tabulate(ratedMovieOfUsers.size) { u =>
		val sum = ratedMovieOfUsers(u)
		            .map{ i => days(times(u)(i), minStamp)}
		            .reduceLeft(_+_)
		if(ratedMovieOfUsers(u).size > 0)
			sum.toDouble / ratedMovieOfUsers(u).size
		else
			globalMeanDate
	}
	//need to fix and the one in SVD++ SVD
	val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	val globalMeanDate = ratingFile.foldLeft(0.0)( (a,b) => a + days(b.timestamp, minStamp)) / ratingFile.size

	def predict(userIndex: Int, movieIndex: Int) = { 
		overallAverage + userDeviation(userIndex) + movieDeviation(movieIndex) + dotProduct(userIndex, movieIndex)

	}

	private def gradientDescent(): Double = {

		for(u <- 0 until numUsers ; i <- 0 until numMovies){
			if (ratings(u)(i) > 0){ //??
				val stamp = times(u)(i)
				val t = days(stamp, minStamp)
				val binT = bin(t)
				val dev_ut = dev(u, t)

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

				//!! lambda
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

	def days(d1: Long, d2: Long) = (TimeUnit.SECONDS.toDays(math.abs(d1 - d2))).toInt
	def bin(day: Int) = (30.0 * day / numDays.toDouble ).toInt
	def dev(u: Int, t: Int) = math.signum(t - userMeanDate(u)) * math.pow(math.abs(t - userMeanDate(u)), beta)
/*
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
*/
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

		val select = 3
		
		//Training
		val matrix = select match {
			case 1 => 
				println("Running matrix fatorization algorithm")
				println
				new MatrixFacotrization
			case 2 => 
				println("Running SVD algorithm")
				println
				new SVD
			case 3 => 
				println("Running SVD++ algorithm")
				println
				new SVDPlus
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
For MovieLen 1m file with factor=2
Matrix : 0.744
SVD : 0.707
SVD++ : 0.743 10 steps
        0.727 50 steps

SVD++

nFB(number of yj)用未知的定義，包含for(i <- 0 until nFB)的用法

(gamma, lambda1, lambda2) = (0.007, 0.005, 0.015)
w : numMovies
nFB = 20
 0.766 5000 steps
 0.851 32740 step - 14735.7 seconds
nFB = 2
 0.755 5000 steps
 0.834 19917 steps (因此不是train越多越好)

nFB = 2
(gamma, lambda1, lambda2) = (0.00007, 0.00005, 0.00015)
w: nFB
 0.738

nFB用原來的定義 i.e. nFB=numMovies 
weight = numMovies 
factor = 2
 0.797 10 steps
 0.745 15 steps
 0.827 20 steps
 0.805 30 steps
 0.778 50 steps
 0.827 100or150 steps
 0.722 100or150 steps
 0.825 200 steps

factor = 10
 0.926 50 steps
 0.856 150 steps

*/		