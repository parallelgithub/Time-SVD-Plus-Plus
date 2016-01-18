import function._

import scala.util.Random
import scala.math

object Main {

	def main(args: Array[String]){

		//挑出每一個user為候選
		//val userCandidate = List.range(1, numUsers+1)

		//挑出所有有兩份評分的user為候選 (取一份評分當test後，至少還留下一份做train)
		//val userCandidate = ratingFile.map{ _.userID }.toSet.toList
		val userCandidate = ratingFile.map{ _.userID }
		                     .groupBy(x => x)
		                     .filter{case(x,y) => y.size > 2}
		                     .map{case(x,y) => x}
		                     .toList

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
			case 5 =>
				println("Running timeSVD++ algorithm")
				println
				new TimeSVDplus

		}

		var mae: Double = 0.0
		var rmse: Double = 0.0
		var evaluateCount: Int = 0

		//Test
		for(test <- testData){

			val actualRating = test.rating
			
			//val predictRating = matrix.predict(test.userID-1, test.movieID - 1)
			val predictRating = matrix.scalePredict(test.userID-1, test.movieID - 1)

			println("User " + test.userID + " with movie " + test.movieID + " : ")
			println(" Predic rating " + "%.3f".format(predictRating) )
			println(" Actual rating " + test.rating)
			println				

			mae += math.abs(actualRating - predictRating)
			rmse += (actualRating - predictRating) * (actualRating - predictRating)
			evaluateCount += 1
		}
		println("File name : " + filePath)
		println("There are " + ratingFile.size + " ratings from " + numUsers + " users on " + numMovies + " movies")
		println("Number of test cases : " + testData.size)
		println("Number of factors : " + numFactors)
		println("MAE = " + "%.3f".format(mae / evaluateCount) )
		println("RMSE = " + "%.3f".format(math.sqrt(rmse / evaluateCount)) )


	} //end of def main()



} // end of object SVD

