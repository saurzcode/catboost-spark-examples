package in.saurzcode
import ai.catboost.spark.{CatBoostClassificationModel, CatBoostClassifier, Pool}
import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
object CatBoostSparkFeatureImportance {
  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("ClassifierTest")
    .getOrCreate();

  val srcDataSchema = Seq(
    StructField("features", SQLDataTypes.VectorType),
    StructField("label", StringType)
  )

  val trainData = Seq(
    Row(Vectors.dense(0.1, 0.2, 0.11), "0"),
    Row(Vectors.dense(0.97, 0.82, 0.33), "1"),
    Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
    Row(Vectors.dense(0.8, 0.62, 0.0), "0")
  )

  val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
  val trainPool = new Pool(trainDf)

  val evalData = Seq(
    Row(Vectors.dense(0.22, 0.33, 0.9), "1"),
    Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
    Row(Vectors.dense(0.77, 0.0, 0.0), "1")
  )

  val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
  val evalPool = new Pool(evalDf)

  val classifier = new CatBoostClassifier

  // train model
  val model = classifier.fit(trainPool, Array[Pool](evalPool))

  // apply model
  val predictions = model.transform(evalPool.data)
  println("predictions")
  predictions.show()

  // save model
  val savedModelPath = "/my_models/binclass_model"
  model.write.save(savedModelPath)

  // save model as local file in CatBoost native format
  val savedNativeModelPath = "./my_local_models/binclass_model.cbm"
  model.saveNativeModel(savedNativeModelPath)

  // load model (can be used in a different Spark session)

  val loadedModel = CatBoostClassificationModel.load(savedModelPath)

  val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
  println("predictionsFromLoadedModel")
  predictionsFromLoadedModel.show()


  // load model as local file in CatBoost native format

  val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

  val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
  println("predictionsFromLoadedNativeModel")
  predictionsFromLoadedNativeModel.show()
}
