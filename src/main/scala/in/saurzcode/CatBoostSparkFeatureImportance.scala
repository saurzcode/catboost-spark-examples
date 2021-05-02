package in.saurzcode

import ai.catboost.spark.CatBoostClassificationModel
import org.apache.spark.sql.SparkSession

object CatBoostSparkFeatureImportance {

  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("FeatureImportanceTest")
      .getOrCreate();

    val loadedModel = CatBoostClassificationModel.loadNativeModel("models/binclass_model.cbm")

    val featureImportance = loadedModel.getFeatureImportancePrettified()

    featureImportance.foreach(fi => println("[" + fi.featureName + "," + fi.importance + "]"))

  }
}
