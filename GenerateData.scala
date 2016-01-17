import java.text.SimpleDateFormat
import java.text.DateFormat
import java.util.Date
import java.sql.Timestamp
import java.util.concurrent.TimeUnit
import java.io.File
import java.io.PrintWriter

import scala.util.Random
import scala.math
import scala.io.Source

object GenerateData {

	//tune
	//!! totalUsers may gap
	val (numUsers, numMovies) = (943, 1682)

	var numRatings = 0
	val (totalUsers, totalMovies) = (2649429, 17770)
	val randomUsers = generateRandomData(numUsers, totalUsers)
	val randomMovies = generateRandomData(numMovies, totalMovies)

def main(args: Array[String]){

	val writer = new PrintWriter(new File("tempRatings.dat"))	

	randomMovies.foreach{ case(oldMovieID, newMovieID) => 

		val fileName = "../dataset/netflix/training_set/mv_" + "%07d".format(oldMovieID) +".txt"
		val (filePath, splitStr) = (fileName, ",")
		val trainFile = Source.fromFile(filePath)		
			.getLines
			.drop(1)			
		    .foreach{line =>
		    	
				val fields = line.split(splitStr)

				val (id, date) = (fields(0).toInt, stringToTimestamp(fields(2)) )
				
				if(randomUsers.contains(id)  ) {
					
					numRatings += 1
					
					val outputLine = randomUsers(id)+ "::"+ newMovieID+"::"+ fields(1)+"::"+ date + "\n"
					
					writer.write(outputLine)

				}


			}				
	}
	writer.close
	println("There are " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")
}

def stampToDays(d: Long) = TimeUnit.MILLISECONDS.toDays(d).toInt
def stringToTimestamp(str: String): Long = {
	val formatter: DateFormat = new SimpleDateFormat("yyyy-MM-dd")
	val date: Date = formatter.parse(str)
	date.getTime()
}

def generateRandomData(numOfNeed: Int, total: Int): collection.mutable.HashMap[Int, Int] = {

		var candidateID = scala.collection.mutable.ListBuffer.range(1, total+1)
		var randomData = collection.mutable.HashMap[Int, Int]()

		var base = total

		for(newID <- 1 to numOfNeed){
				val i = Random.nextInt(base)	
				randomData += (candidateID(i) -> newID)
				candidateID.remove(i)
				base -= 1
		}	

		randomData
}

}