package in.saurzcode.classification

import ai.catboost.spark.{CatBoostClassificationModel, CatBoostClassifier, Pool}
import in.saurzcode.spark.SparkSessionWrapper
import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}

object CatBoostSparkBinaryClassification extends SparkSessionWrapper {

  def main(args: Array[String]): Unit = {


    val srcDataSchema = Seq(
      StructField("features", SQLDataTypes.VectorType),
      StructField("label", StringType)
    )

    val trainData = Seq(
      Row(Vectors.dense(0.11, 0.22, 0.13, 0.45, 0.89), "0"),
      Row(Vectors.dense(0.99, 0.82, 0.33, 0.89, 0.97), "1"),
      Row(Vectors.dense(0.12, 0.21, 0.23, 0.42, 0.24), "1"),
      Row(Vectors.dense(0.81, 0.63, 0.02, 0.55, 0.65), "0")
    )

    val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
    val trainPool = new Pool(trainDf)

    val evalData = Seq(
      Row(Vectors.dense(0.22, 0.34, 0.9, 0.66, 0.99), "1"),
      Row(Vectors.dense(0.16, 0.1, 0.21, 0.67, 0.46), "0"),
      Row(Vectors.dense(0.78, 0.0, 0.0, 0.22, 0.12), "1")
    )

    val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
    val evalPool = new Pool(evalDf)

    val classifier = new CatBoostClassifier

    // train model
    val model: CatBoostClassificationModel = classifier.fit(trainPool, Array[Pool](evalPool))

    // apply model
    val predictions: DataFrame = model.transform(evalPool.data)
    println("predictions")
    predictions.show(false)

    // save model
    val savedModelPath = "models/binclass_model"
    model.write.overwrite().save(savedModelPath)

    // save model as local file in CatBoost native format
    val savedNativeModelPath = "models/binclass_model.cbm"
    model.saveNativeModel(savedNativeModelPath)

    // load model (can be used in a different Spark session)

    val loadedModel = CatBoostClassificationModel.load(savedModelPath)

    val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
    println("predictionsFromLoadedModel")
    predictionsFromLoadedModel.show(false)

    // load model as local file in CatBoost native format

    val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

    val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
    println("predictionsFromLoadedNativeModel")
    predictionsFromLoadedNativeModel.show(false)

    spark.stop()
  }

}
