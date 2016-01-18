package function
import function._

import scala.util.Random
import scala.math
import java.util.concurrent.TimeUnit

class TimeSVD extends TrainingModel {

	val (beta) = (0.3)
	val (gamma, lambda) = (0.0002, 0.2)

	val numBins = 30
	//b_u
	val userDeviation = Array.fill(numUsers)(Random.nextDouble)
	//b_u
	val userDeviationT = Array.fill(numUsers)(collection.mutable.HashMap[Int, Double]())
	ratingFile.foreach{ x => 
		val t = days(x.timestamp, minStamp)
		userDeviationT(x.userID - 1) += (t -> 0.0) 
	}
	
	//b_i
	val movieDeviation = Array.fill(numMovies)(Random.nextDouble)
	//b_i-bin(t)
	val movieDeviationT = Array.fill(numMovies)(Array.fill(numBins)(Random.nextDouble))
	//alpha_u
	val alpha = Array.fill(numUsers)(Random.nextDouble)
	//alpahK_u
	val alphaK = Array.fill(numUsers)(Array.fill(numFactors)(Random.nextDouble))
	//p_u(t)
	val timePreference = Array.fill(numUsers)( Array.fill(numFactors)(collection.mutable.HashMap[Int, Double]()) )

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
		            .foldLeft(0.0)(_+_)
		if(ratedMovieOfUsers(u).size > 0)
			sum / ratedMovieOfUsers(u).size
		else
			globalMeanDate
	}

	//mui
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
		val bUT = if (userDeviationT(userIndex).contains(t)) userDeviationT(userIndex)(t) else 0.0

		var sum = 0.0
		for(k <- 0 until numFactors) {
			val put = if (timePreference(userIndex)(k).contains(t)) 
			             timePreference(userIndex)(k)(t)
			           else 
			             0.0

			//q_i * ( p_u + alpha_u * dev_u(t) + p_u(t) )
			sum += matrixQ(k)(movieIndex) * 
			         (matrixP(userIndex)(k) + 
			          alphaK(userIndex)(k) * dev(userIndex, t) + 
			          put)
		}	

		//!! check bui-contain
		//prediction = mui + b_u + alaph_u * dev_u(t) + b_ut + b_i + b_i-bin(t) + q_i * ( p_u + alpha_u * dev_u(t) + p_u(t))
		overallAverage + 
		  userDeviation(userIndex) + alpha(userIndex) * dev(userIndex, t) + bUT +
		  movieDeviation(movieIndex) + movieDeviationT(movieIndex)(binT) + sum

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

				val eui = ratings(u)(i) - predict(u,i)
				val but = userDeviationT(u)(t) + 
				          gamma * (eui - lambda * userDeviationT(u)(t))

				//update b_u
				userDeviation(u) += gamma * (eui - lambda * userDeviation(u))
				//update alpha_u
				alpha(u) += gamma * (eui * dev(u, t) - lambda * alpha(u))
				//update b_ut
				userDeviationT(u) += (t -> but)
				//update b_i
				movieDeviation(i) += gamma * (eui - lambda * movieDeviation(i))
				//update b_i-bin(t)
				movieDeviationT(i)(binT) += gamma * (eui - lambda * bit)

				for(k <- 0 until numFactors){
					val pu = matrixP(u)(k)
					val qi = matrixQ(k)(i)
					val alphaU = alphaK(u)(k)

					if(!timePreference(u)(k).contains(t))
						timePreference(u)(k) += (t -> 0.0)
					val put = timePreference(u)(k)(t) + 
					           gamma * (eui * qi - lambda * timePreference(u)(k)(t))

					//update p_u
					matrixP(u)(k) += gamma * ( eui * qi - lambda * pu)
					//update q_i
					matrixQ(k)(i) += gamma * ( eui * (pu + alphaU * dev(u,t) + put) - lambda * qi )
					//update alpha_uk
					alphaK(u)(k) += gamma * ( eui * qi * dev(u, t) - lambda * alphaU)
					//update p_ku(t)
					timePreference(u)(k) += (t -> put)
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

				for(k <- 0 until numFactors){
					val auk = alphaK(u)(k)
					val pu = matrixP(u)(k)
					val qi = matrixQ(k)(i)
					val put = if (timePreference(u)(k).contains(t)) 
					             timePreference(u)(k)(t) 
					           else 
					             0.0
					
					error += (lambda/2.0) * ( auk * auk + pu * pu + qi * qi + put * put)
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
