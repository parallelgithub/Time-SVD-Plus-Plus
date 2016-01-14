

import scala.io.Source
import scala.util.Random
import scala.math
import java.util.concurrent.TimeUnit

		case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

		val (filePath, splitStr) = ("../dataset/ml-100k/u.data", "\t")
		//val (filePath, splitStr) = ("../dataset/ml-1m/ratings.dat", "::")
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

	val steps = 1000

	val (gamma, lambda) = (0.002, 0.02)
	//(0.001, 0.02) 
	//(0.01, 0.05) http://blog.csdn.net/zhaoxinfan/article/details/8821419
	//(0.015, 0.015) 2008 (SVD) A Guide to Singular Value Decomposition for Collaborative Filtering

	val overallAverage = {
		var count = 0
		var sum = 0.0
		for(u <- 0 until numUsers ; i <- 0 until numMovies)
			if(ratings(u)(i) > 0.0){
				sum += ratings(u)(i)
				count += 1
			}
		sum / count
	}

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

//https://github.com/guoguibing/librec/blob/master/librec/src/main/java/librec/rating/SVDPlusPlus.java
//https://www.quora.com/Whats-the-difference-between-SVD-and-SVD++
//"Factorization meets the neighborhood- a multifaceted collaborative filtering model", 2008
class SVDPlus extends TrainingModel {

	val nFB = numMovies 
	def w(value: Double): Double = value * 1000.0

	val steps = 50
	val (gamma, lambda1, lambda2) = (0.007, 0.005, 0.015)
	//(0.007, 0.005, 0.015) "Factorization meets the neighborhood- a multifaceted collaborative filtering model"

	val overallAverage = {
		var count = 0
		var sum = 0.0
		for(u <- 0 until numUsers ; i <- 0 until numMovies)
			if(ratings(u)(i) > 0.0){
				sum += ratings(u)(i)
				count += 1
			}
		sum / count
	}

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

	val steps = 30

	val (beta) = (0.4)
	val (gamma, lambda) = (0.002, 0.01)

	//!! how to init
	val numBins = 30
	val userDeviation = Array.fill(numUsers)(Random.nextDouble)
	val movieDeviation = Array.fill(numMovies)(Random.nextDouble)
	val movieDeviationT = Array.fill(numMovies)(Array.fill(numBins)(Random.nextDouble))
	val alpha = Array.fill(numUsers)(Random.nextDouble)
	val ratedMovieOfUsers = ratings.map{ x => 
		                        val s = x.size 
		                        for(i <- 0 until s if x(i) > 0.0) 
		                        	yield i 
		                        }

	val minStamp = ratingFile.reduceLeft( (a,b) => if (a.timestamp < b.timestamp) a else b).timestamp
	val maxStamp = ratingFile.reduceLeft( (a,b) => if (a.timestamp > b.timestamp) a else b).timestamp
	val numDays = days(maxStamp, minStamp) + 1
	val times = Array.fill(numUsers)(Array.fill[Long](numMovies)(0))
	//For the rating(u)(i) which we want to predict(to test), its timestamp is preserved
	//另一種可能的作法是都設為現在的時間
	ratingFile.foreach{ x => times(x.userID - 1)(x.movieID - 1) =  x.timestamp }
	val userMeanDate = Array.tabulate(ratedMovieOfUsers.size) { u =>
		val sum: Double = ratedMovieOfUsers(u)
		            .map{ i => days(times(u)(i), minStamp)}
		            .reduceLeft(_+_)
		if(ratedMovieOfUsers(u).size > 0)
			sum / ratedMovieOfUsers(u).size
		else
			globalMeanDate
	}
	//need to fix and the one in SVD++ SVD
	//val overallAverage = ratingFile.foldLeft(0.0)( (a,b) => a + b.rating) / ratingFile.size
	val overallAverage = {
		var count = 0
		var sum = 0.0
		for(u <- 0 until numUsers ; i <- 0 until numMovies)
			if(ratings(u)(i) > 0.0){
				sum += ratings(u)(i)
				count += 1
			}
		sum / count
	}	
	//val globalMeanDate = ratingFile.foldLeft(0.0)( (a,b) => a + days(b.timestamp, minStamp)) / ratingFile.size
	val globalMeanDate = {
		var count = 0
		var sum = 0.0
		for(u <- 0 until numUsers ; i <- 0 until numMovies)
			if(ratings(u)(i) > 0.0){
				sum += days(times(u)(i), minStamp)
				count += 1
			}
		sum / count
	}	

	def predict(userIndex: Int, movieIndex: Int) = { 
		val stamp = times(userIndex)(movieIndex)
		val t = days(stamp, minStamp)
		val binT = bin(t)
		//!! check bui-contain
		overallAverage + 
		  userDeviation(userIndex) + alpha(userIndex) * dev(userIndex, t)
		  movieDeviation(movieIndex) + movieDeviationT(movieIndex)(binT)
		  dotProduct(userIndex, movieIndex)

	}

	private def gradientDescent(): Double = {

		for(u <- 0 until numUsers ; i <- 0 until numMovies){
			if (ratings(u)(i) > 0){ //??
				val stamp = times(u)(i)
				val t = days(stamp, minStamp)
				val binT = bin(t)
				val devUT = dev(u, t)

				val bi = movieDeviation(i)
				val bit = movieDeviationT(i)(binT)
				val bu = userDeviation(u)
				//val but = userDeviationT(u, t)

				val au = alpha(u)

				//equation (11)
				val bui = overallAverage + bu + au * devUT + /*but +*/ bi + bit

				val eui = ratings(u)(i) - predict(u,i)

				//update
				userDeviation(u) += gamma * (eui - lambda * userDeviation(u))
				movieDeviation(i) += gamma * (eui - lambda * movieDeviation(i))
				movieDeviationT(i)(binT) += gamma * (eui - lambda * bit)
				alpha(u) += gamma * (eui * devUT - lambda * au)

				for(f <- 0 until numFactors){
					val puf = matrixP(u)(f)
					matrixP(u)(f) += gamma * ( eui * matrixQ(f)(i) - lambda * matrixP(u)(f))
					matrixQ(f)(i) += gamma * ( eui * puf - lambda * matrixQ(f)(i))
				}
			}
		}
		
		var error = 0.0
		for(u <- 0 until numUsers; i <- 0 until numMovies){
			if (ratings(u)(i) > 0){ //??
				val eui = ratings(u)(i) - predict(u,i)
				error += eui * eui

				//!! lambda
				val bu = userDeviation(u)
				val bi = movieDeviation(i)
				error += (lambda/2.0) * ( bu * bu + bi * bi )

				for(f <- 0 until numFactors){					
					val pu = matrixP(u)(f)
					val qi = matrixQ(f)(i)
					
					error += (lambda/2.0) * ( pu * pu + qi * qi )
				}
			}
		}
		error  
	}

	def days(d1: Long, d2: Long) = (TimeUnit.SECONDS.toDays(math.abs(d1 - d2))).toInt
	def bin(day: Int) = (numBins.toDouble * day / numDays.toDouble ).toInt
	def dev(u: Int, t: Int) = math.signum(t - userMeanDate(u)) * math.pow(math.abs(t - userMeanDate(u)), beta)

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

		val select = 4
		
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
			case 4 =>
				println("Running timeSVD algorithm")
				println
				new TimeSVD

		}

		var mae: Double = 0.0
		var rmse: Double = 0.0
		var evaluateCount: Int = 0

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

			mae += math.abs(actualRating - predictRating)
			rmse += (actualRating - predictRating) * (actualRating - predictRating)
			evaluateCount += 1
		}
		println("File name : " + filePath)
		println("Number of factors : " + numFactors)
		println("MAE = " + "%.3f".format(mae / evaluateCount) )
		println("RMSE = " + "%.3f".format(math.sqrt(rmse / evaluateCount)) )

/*
(MAE, RMSE)

**For MovieLen 1m file with factor=2
[Matrix] : (0.744, )
[SVD] : (0.691, ) 1000 steps
[SVD++] : (0.753, 0.965) 10 steps
          0.727 50 steps


**For MovieLen 100k file 
[SVD]
factor = 2
 0.723 315 steps
factor = 10
 0.841 100 steps
 0.834 500 steps
 0.872 1000 steps
 0.912 5000 steps
[SVD++]
factor = 2
 0.825 5 steps
 0.866 10 steps
 0.795 15 steps
 0.766 20 steps
 ***
 0.805 30 steps
 0.778 50 steps
 0.827 100or150 steps
 0.722 100or150 steps
 0.825 200 steps
factor = 10
 0.788 15 steps
  50 steps
  150 steps
[timeSVD - part] compare to the paper Table2
factor = 2
 (0.822, 1.062) 20 steps
 (0.748, 0.981) 30 steps
 (0.767, 0.959) 50 steps
 (0.746, 0.954) 100 steps
 (0.786, 1.015) 200 steps
 (0.793, 0.994) 500 steps
 (0.842, 1.053) 1000 steps
 () steps
factor = 10
 (0.852, 1.089) 20 steps
 (0.813, 1.040) 25 steps
 (0.780, 1.032) 30 steps
factor = 20 
 (0.858, 1.144) 20 steps
 (0.839, 1.083) 25 steps
 (0.827, 1.082) 30 steps
factor = 50
 (1.073, 1.507) 20 steps
 (1.041, 1.426) 25 steps
 (1.008, 1.314) 30 steps
factor = 100
 (1.110, 1.660) 20 steps
 (1.216, 1.769) 25 steps
 (1.287, 2.020) 30 steps
facetor = 200
 (1.815, 3.388) 20 steps
 (1.799, 3.567) 30 steps
 (0.987, 1.360) 30 steps - 1m dataset
*/		