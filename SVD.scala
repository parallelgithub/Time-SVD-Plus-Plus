
import scala.io.Source
import scala.util.Random
import scala.math
import java.util.concurrent.TimeUnit

object SVD {

	case class RatingDataStructure(userID: Int, movieID: Int, rating: Double, timestamp: Long)

	//set algorithm, number of trainging iteration, 
	//and number of factors in matrix factorization
	val (selectAlgorithm, steps, numFactors) = (4, 10, 2)
	
	val (filePath, splitStr) = ("ratingsNetflix.dat", "::") //從Netflix篩選出的小檔案
	//val (filePath, splitStr) = ("ratings20m.dat", "::") //從MovieLen 20m 篩選出的小檔案
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

	def main(args: Array[String]){

		//挑出每一個user為候選
		//val userCandidate = List.range(1, numUsers+1)

		//挑出所有有評分的user為候選
		val userCandidate = ratingFile.map{ _.userID }.toSet.toList

		//隨機從候選users中挑出testSize個test users
		val testSize = (0.3 * userCandidate.size).toInt

		//從有n個候選user的candidate中隨機選出count個test users
		def generateTestUsers(candidate: List[Int], n: Int, count: Int): List[Int] = {
			if(count == 0)
				Nil
			else{
				val i = Random.nextInt(n)	
				candidate(i) :: generateTestUsers((candidate.take(i) ::: candidate.drop(i+1)), n-1, count-1)
			}
		}
		//儲存 test users 的資料，並將預計測試的電影在matrix重設為0
		val testData = generateTestUsers(userCandidate, userCandidate.size, testSize).map{ id =>
			val recent = ratingFile 
			              .filter( _.userID == id ) 
			              .reduceLeft( (a,b) => if (a.timestamp > b.timestamp) a else b)
					
			val actualRating = ratings(id - 1)(recent.movieID - 1)
			ratings(id - 1)(recent.movieID - 1) = 0.0
			RatingDataStructure(id, recent.movieID, actualRating, recent.timestamp )
		}
		
		//Training
		val matrix = selectAlgorithm match {
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
				new SVDplus
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


	} //end of def main()

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

	} //end of class MatrixFactorization

	//http://blog.csdn.net/zhaoxinfan/article/details/8821419
	//http://sifter.org/~simon/journal/20061211.html
	//"Matrix factorization techniques for recommender systems", 2009 
	class SVD extends TrainingModel {

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

	} //end of class SVD

	//https://github.com/guoguibing/librec/blob/master/librec/src/main/java/librec/rating/SVDPlusPlus.java
	//https://www.quora.com/Whats-the-difference-between-SVD-and-SVD++
	//"Factorization meets the neighborhood- a multifaceted collaborative filtering model", 2008
	class SVDplus extends TrainingModel {

		val nFB = numMovies 
		def w(value: Double): Double = value * 1000.0

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
	} //end of class SVDplus

	class TimeSVD extends TrainingModel {

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
		val userDeviationT = Array.fill(numUsers)(collection.mutable.HashMap[Int, Double]())
		ratingFile.foreach{ x => 
			val t = days(x.timestamp, minStamp)
			userDeviationT(x.userID - 1) += (t -> 0.0) 
		}
		//userDeviationT.foreach{println}

		val userMeanDate = Array.tabulate(ratedMovieOfUsers.size) { u =>
			val sum: Double = ratedMovieOfUsers(u)
			            .map{ i => days(times(u)(i), minStamp)}
			            .foldLeft(0.0)(_+_)
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
			val but = if (userDeviationT(userIndex).contains(t)) userDeviationT(userIndex)(t) else 0.0
			//!! check bui-contain
			overallAverage + 
			  userDeviation(userIndex) + alpha(userIndex) * dev(userIndex, t) + but +
			  movieDeviation(movieIndex) + movieDeviationT(movieIndex)(binT) +
			  dotProduct(userIndex, movieIndex)

		}

		private def gradientDescent(): Double = {

			for(u <- 0 until numUsers ; i <- 0 until numMovies){
				if (ratings(u)(i) > 0){ //??

					//唯有ratings(u)(i) > 0 時，stamp才不為0、t才會合理
					val stamp = times(u)(i)
					val t = days(stamp, minStamp)
					val binT = bin(t)
					val bit = movieDeviationT(i)(binT)

					if(!userDeviationT(u).contains(t))
						userDeviationT(u) += (t -> 0.0)
					val but = userDeviationT(u)(t)

					//equation (11)
					//val bui = overallAverage + bu + au * devUT + but + bi + bit

					val eui = ratings(u)(i) - predict(u,i)
					val butNew = but + gamma * (eui - lambda * but)

					//update
					userDeviation(u) += gamma * (eui - lambda * userDeviation(u))
					userDeviationT(u) += (t -> butNew)
					movieDeviation(i) += gamma * (eui - lambda * movieDeviation(i))
					movieDeviationT(i)(binT) += gamma * (eui - lambda * bit)
					alpha(u) += gamma * (eui * dev(u, t) - lambda * alpha(u))

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

					val stamp = times(u)(i)
					val t = days(stamp, minStamp)
					val binT = bin(t)
					val bit = movieDeviationT(i)(binT)
					val bu = userDeviation(u)
					val au = alpha(u)

					val but = if(userDeviationT(u).contains(t)) userDeviationT(u)(t) else 0.0

					val bi = movieDeviation(i)
					error += (lambda/2.0) * 
					         ( bu * bu + au * au + but * but + bi * bi + bit * bit)

					for(f <- 0 until numFactors){					
						val pu = matrixP(u)(f)
						val qi = matrixQ(f)(i)
						
						error += (lambda/2.0) * ( pu * pu + qi * qi )
					}
				}
			}
			error  
		}

		//def days(d1: Long, d2: Long) = (TimeUnit.SECONDS.toDays(math.abs(d1 - d2))).toInt
		def days(d1: Long, d2: Long) = (TimeUnit.MILLISECONDS.toDays(math.abs(d1 - d2))).toInt
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

	} //end of class TimeSVD


} // end of object TimeSVDplus

/*
(MAE, RMSE)

**For MovieLen 1m file with factor=2
[Matrix] : 
[SVD] :  1000 steps
[SVD++] :  10 steps
           50 steps

**For MovieLen 100k file 

[Matrix factorization]
factor = 2
 (0.792, 1.034) 100 steps
 (0.778, 1.014) 500 steps
factor = 100
 (1.354, 1.854) 100 steps

[SVD]
factor = 2
 (0.800, 1.035) 100 steps
 (0.824, 1.091) 200 steps
 (0.764, 1.014) 300 steps
 0.723 315 steps
factor = 10
 (0.835, 1.071) 100 steps

[SVD++]
factor = 2
 (0.846, 1.067) 5 steps
 (0.819, 1.062) 10 steps
 (0.841, 1.052) 20 steps
 0.722 100or150 steps
factor = 10
 (0.886, 1.139) 5 steps
 0.788 15 steps

[timeSVD - part] compare to the paper Table2
factor = 2
 (0.798, 1.047) 20 steps
 (0.752, 0.983) 30 steps
 (0.776, 0.994) 100 steps
factor = 10
 (0.821, 1.054) 20 steps
 (0.868, 1.109) 25 steps
 (0.818, 1.039) 30 steps
factor = 20 
 (0.928, 1.189) 20 steps
 (0.849, 1.141) 30 steps
factor = 50
 (1.125, 1.603) 20 steps
 (1.085, 1.462) 30 steps
factor = 100
 (1.1396, 1.909) 20 steps
 (1.308, 1.929) 30 steps
facetor = 200
 (1.619, 2.801) 20 steps
 (1.522, 2.664) 30 steps
 (1.518, 2.439) 50 steps

*/		