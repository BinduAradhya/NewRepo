# bindu aradhya janagani bj266

# Importing necessary modules
from pyspark.sql import SparkSession  # For creating and managing Spark sessions
from pyspark.ml import PipelineModel  # To load the saved ML pipeline model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # To evaluate the model's performance
import sys  # For command-line argument parsing
from pyspark.ml.feature import StringIndexer  # To convert string labels to numerical indices
from pyspark.ml.feature import VectorAssembler  # To assemble multiple feature columns into a single vector column
import os  # For handling file paths

def prepare_data(input_data):
    """
    Prepares the input data for prediction by cleaning column names, indexing the label column, 
    and assembling feature columns into a single 'features' column.

    Parameters:
    input_data (DataFrame): The input dataset loaded as a Spark DataFrame.

    Returns:
    DataFrame: The prepared dataset with a 'features' column and a numerical 'label' column.
    """
    # Remove quotes from column names if present
    new_columns = [col.replace('"', '') for col in input_data.columns]
    input_data = input_data.toDF(*new_columns)

    # Define the column that represents the label
    label_column = 'quality'

    # Convert the categorical 'quality' column to numerical indices
    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    # Identify all columns except the label column for feature extraction
    feature_columns = [col for col in input_data.columns if col != label_column]

    # Assemble all feature columns into a single vector column called 'features'
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Transform the input data to include the 'features' column
    assembled_data = assembler.transform(input_data)

    return assembled_data

def predict_using_model(test_data_path, output_model):
    """
    Loads a trained ML model, prepares the test data, and evaluates the model's predictions.

    Parameters:
    test_data_path (str): Path to the CSV file containing the test dataset.
    output_model (str): Path to the directory containing the trained ML model.
    """
    # Create a Spark session for distributed data processing
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load the raw test data from the CSV file
    test_raw_data = spark.read.csv(test_data_path, header=True, inferSchema=True, sep=";")

    # Prepare the test data by processing it into a suitable format
    test_data = prepare_data(test_raw_data)

    # Load the trained ML pipeline model from the specified path
    model_path = os.path.join(os.getcwd(), output_model)
    trained_model = PipelineModel.load(model_path)

    # Generate predictions using the test data
    predictions = trained_model.transform(test_data)

    # Initialize the evaluator for classification metrics
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Calculate the accuracy of the predictions
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    # Calculate the F1 score of the predictions
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    # Print the evaluation results
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1_score}")

    # Stop the Spark session to free up resources
    spark.stop()

if __name__ == "__main__":
    """
    Main entry point for the script. Expects two command-line arguments:
    1. Path to the test data CSV file.
    2. Path to the directory containing the trained model.
    """
    # Ensure the script is invoked with exactly two arguments
    if len(sys.argv) != 3:
        print("Usage: spark-submit main.py <test_data_path> <output_model>")
        sys.exit(1)

    # Read the command-line arguments
    test_data_path = sys.argv[1]
    output_model = sys.argv[2]

    # Call the function to load the model, make predictions, and evaluate
    predict_using_model(test_data_path, output_model)

# bindu aradhya janagani bj266
