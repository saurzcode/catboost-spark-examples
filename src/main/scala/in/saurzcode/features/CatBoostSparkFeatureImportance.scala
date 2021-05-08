package in.saurzcode.features

import ai.catboost.spark.CatBoostClassificationModel
import in.saurzcode.spark.SparkSessionWrapper

object CatBoostSparkFeatureImportance extends SparkSessionWrapper{

  def main(args: Array[String]): Unit = {

    val loadedModel = CatBoostClassificationModel.loadNativeModel("models/binclass_model.cbm")

    val featureImportance = loadedModel.getFeatureImportancePrettified()

    featureImportance.foreach(fi => println("[" + fi.featureName + "," + fi.importance + "]"))

  }
}
