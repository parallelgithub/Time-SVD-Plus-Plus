package function
import function._

import scala.util.Random

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

	def scalePredict(userIndex: Int, movieIndex: Int) = {

		val value = predict(userIndex, movieIndex)

		value match {
			case v if v < 1.0 => 1.0
			case v if v > 5.0 => 5.0
			case _ => value
		}			

	}
	def predict(userIndex: Int, movieIndex: Int): Double 
}
