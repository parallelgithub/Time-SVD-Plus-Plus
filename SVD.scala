package function
import function._

import scala.math

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
