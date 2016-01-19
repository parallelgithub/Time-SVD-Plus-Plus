package function
import function._

import scala.util.Random
import scala.math

//https://github.com/guoguibing/librec/blob/master/librec/src/main/java/librec/rating/SVDPlusPlus.java
//https://www.quora.com/Whats-the-difference-between-SVD-and-SVD++
//"Factorization meets the neighborhood- a multifaceted collaborative filtering model", 2008
class SVDplus extends TrainingModel {

	val nFB = numMovies 
	def w(value: Double): Double = math.sqrt(value) 

	//Good
	val (gamma, lambda1, lambda2) = (0.004, 0.02, 0.03)
	//(0.002, 0.04, 0.04) "Factorization meets the neighborhood- a multifaceted collaborative filtering model"

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
				value += feedback(f)(j) * matrixQ(f)(movieIndex)

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
					for(j <- ratedMovieOfUsers(u)){
						feedback(f)(j) += gamma * ( eui * matrixQ(f)(i) / w(ratedMovieOfUsers(u).size) - lambda2 * feedback(f)(j))
						feedback(f)(j) = if (feedback(f)(j) > 0.5) 1 else 0
					}
				}
			}
		} // end of updating

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
	println
	println("min error : " + min + " at step " + step)
	println

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
