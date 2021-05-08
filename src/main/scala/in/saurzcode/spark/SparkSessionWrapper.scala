package in.saurzcode.spark

import org.apache.spark.sql.SparkSession

class SparkSessionWrapper {
  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("CatBoostTest")
    .getOrCreate();
}
