package ak.mll.mulan.bpmll;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.neural.BPMLL;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import ak.mll.mulan.Utility;
import ak.mll.mulan.Utility.PredictionResults;

/**
 * A binary wrapper for MULAN's BPMLL algorithm.
 * When not used as a binary, {@link process} may be used as an API method.
 * API Call examples:
		process("/Users/abhishek/Workspaces/MLL/data/scene/scene-train.arff",
				"/Users/abhishek/Workspaces/MLL/data/scene/scene-test.arff",
				"/Users/abhishek/Workspaces/MLL/data/scene/scene.xml", 
				"/Users/abhishek/Workspaces/MLL/data/scene/predictions.csv",
				4);
		process("/Users/abhishek/Workspaces/MLL/data/yeast/yeast-train.arff",
				"/Users/abhishek/Workspaces/MLL/data/yeast/yeast-test.arff",
				"/Users/abhishek/Workspaces/MLL/data/yeast/yeast.xml", 
				"/Users/abhishek/Workspaces/MLL/data/yeast/predictions.csv",
				6);
		process("/Users/abhishek/Workspaces/MLL/data/emotions/emotions-train.arff",
				"/Users/abhishek/Workspaces/MLL/data/emotions/emotions-test.arff",
				"/Users/abhishek/Workspaces/MLL/data/emotions/emotions.xml", 
				"/Users/abhishek/Workspaces/MLL/data/emotions/predictions.csv",
				4);
		process("/Users/abhishek/Workspaces/MLL/data/enron/enron-train.arff",
				"/Users/abhishek/Workspaces/MLL/data/enron/enron-test.arff",
				"/Users/abhishek/Workspaces/MLL/data/enron/enron.xml", 
				"/Users/abhishek/Workspaces/MLL/data/enron/predictions.csv",
				10);
		process("/Users/abhishek/Workspaces/MLL/data/moviegenre/moviegenre.arff",
				"/Users/abhishek/Workspaces/MLL/data/moviegenre/moviegenre.arff",
				"/Users/abhishek/Workspaces/MLL/data/moviegenre/moviegenre.xml", 
				"/Users/abhishek/Workspaces/MLL/data/moviegenre/predictions.csv",
				10);
 * @author Abhishek (abhishek.kumar.ak@gmail.com)
 *
 */
public class BPMLLBinary {
	
	private static Logger logger = Logger.getLogger(BPMLLBinary.class.getCanonicalName());
	
	// Fixed learning rate for BPMLL as discussed in Zhang's paper.
	private static final double LEARNING_RATE = 0.05;
	
	private static final int EPOCHS = 1000;
	
	/**
	 * Main entry point. Trains a model on the training set, and makes predictions for the
	 * test set.
	 * @param args 4 arguments expected. 'train', 'test', 'xml' and 'output' for the
	 * training file, test file, xml header file, and output file locations respectively.
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		logger.setLevel(Level.FINE);
		
		// Read flags
		String trainingFile = Utils.getOption("train", args);
		String testFile = Utils.getOption("test", args);
		String xmlFile= Utils.getOption("xml", args);
		String outputFile = Utils.getOption("output", args);
		String job = Utils.getOption("options", args);
		
		if ("tune".equals(job)) {
			PredictionResults results = tuneParamsAndCrossValidate(
					Utility.mergeTwoDatasets(trainingFile, testFile, xmlFile).getDataSet(), 
					xmlFile, 10, 10);
			results.writeToDirectory(outputFile);
		} else if ("cv".equals(job)) {
			PredictionResults results = crossValidate(
					Utility.mergeTwoDatasets(trainingFile, testFile, xmlFile).getDataSet(), 
					xmlFile, 10, 10, null);
			results.writeToDirectory(outputFile);
		} else if ("test".equals(job)) {
			PredictionResults results = process(trainingFile, testFile, xmlFile, 10, null);
			results.writeToDirectory(outputFile);
		} else {
			System.err.println("Usage: BPMLLBinary -train=<trainfile path> -test=<test file path> " +
					"-xml=<XML label data file path> -output=<output directory> -options=<tune|cv|test>");
			System.err.print("Options:\n\ttune: tune regularization weight\n" +
					"\tCV: do cross validation\n\ttest:train on training and predict on test set.");
			System.exit(1);
		}
	}
	
	/**
	 * Use the MULAN neural network model with one hidden layer to make predictions
	 * for a multilabel data set.
	 * @param trainingFile the full file path of the training file in ARFF format.
	 * @param testFile the full file path of the test file in ARFF format.
	 * @param xmlFile the full file path of the xml file describing the labels.
	 * @param numHiddenUnits number of hidden units to use in the model. A 
	 * {@code null} value indicates num hidden units = 20% of input space (# features)
	 * @return Predicted results in CSV, serialized to a {@String}. 
	 * @throws Exception
	 */
	protected static PredictionResults process(String trainingFile, String testFile, 
			String xmlFile, Integer numHiddenUnits, Double regularizationWeight) throws Exception {
		logger.info("Processing dataset: " + trainingFile + 
				"\n with " + numHiddenUnits + " hidden units.");
		
		// Read and initialize data sets
		MultiLabelInstances dataset = new MultiLabelInstances(trainingFile, xmlFile);
		MultiLabelInstances testset = new MultiLabelInstances(testFile, xmlFile);
		
		// Train
		BPMLL bpmll = train(dataset, numHiddenUnits, regularizationWeight);
		
		// Predict and return results
		return predict(bpmll, testset);
	}
	
	/**
	 * Use the Mulan BPMLL model to learn a classifier based on a given dataset, and
	 * return predictions on test folds.
	 * @param alldata the dataset to use. It is split into multiple folds for cross
	 * validation.
	 * @param xmlFile XML header file with label details.
	 * @param folds the number of cross validation folds to use.
	 * @param numHiddenUnits number of hidden units to use in model.
	 * @return a pair of prediction results: {prediction, ground truth}
	 * @throws Exception 
	 */
	protected static PredictionResults tuneParamsAndCrossValidate(Instances dataset,
			String xmlFile, int folds, Integer numHiddenUnits) 
					throws Exception {
		
	
		dataset.randomize(new Random(System.currentTimeMillis()));
		
		double bestLL = Integer.MIN_VALUE;
		PredictionResults bestResults = null;
		double prevLL = 0, prevprevLL = 0;
		
		// Tune regularization weight
		for (double regularizationWeight : 
				new double[]{1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8, 1e10}) {
			PredictionResults results = 
					crossValidate(dataset, xmlFile, folds, numHiddenUnits, regularizationWeight);
			double LL = results.getLogLikelihood();
			
			logger.fine("\t\tRegularization Weight: " + regularizationWeight + "; Score: " + LL);
			if (LL >= bestLL) {
				bestLL = LL;
				bestResults = results;
				logger.info("\tBest Regularization Weight so far: " + regularizationWeight);
			}
			
			
			if (LL < prevLL && prevLL < prevprevLL) {
				logger.info("\tBreaking at Regularization Weight: " + regularizationWeight);
				break;
			}
			prevprevLL = prevLL;
			prevLL = LL;
		}
		
		return bestResults;
	}
	
	protected static PredictionResults crossValidate(Instances dataset,
			String xmlFile, int folds, Integer numHiddenUnits, Double regularizationWeight) 
					throws Exception {
		
		PredictionResults results = null;
				
		// Cross Validate
		for (int f = 0; f < folds; f++) {
			MultiLabelInstances trainData = 
					new MultiLabelInstances(dataset.trainCV(folds, f), xmlFile);
			MultiLabelInstances testData = 
					new MultiLabelInstances(dataset.testCV(folds, f), xmlFile);
			
			// Train a BPMLL Model
			BPMLL bpmll = train(trainData, numHiddenUnits, regularizationWeight);
			
			// Record results
			results = PredictionResults.add(results, predict(bpmll, testData));
		}
		
		return results;
	}
	
	/**
	 * Trains a BPMLL model on given data and parameters.
	 * @param trainData Training dataset.
	 * @param numHiddenUnits number of hidden units to use. 
	 * {@code null} implies default value (20% of input space).
	 * @param regularizationWeight {@code null} implies default value of 1e-5.
	 * @return
	 * @throws Exception
	 */
	public static BPMLL train(MultiLabelInstances trainData, Integer numHiddenUnits, 
			Double regularizationWeight) throws Exception {
		BPMLL bpmll = new BPMLL();
		bpmll.setTrainingEpochs(EPOCHS);
		bpmll.setLearningRate(LEARNING_RATE);
		if (numHiddenUnits != null) {
			bpmll.setHiddenLayers(new int[]{numHiddenUnits});
		}
		if (regularizationWeight != null) {
			bpmll.setWeightsDecayRegularization(regularizationWeight);
		}
		bpmll.build(trainData);
		return bpmll;
	}
	
	public static PredictionResults predict(BPMLL bpmll, MultiLabelInstances testSet) 
			throws InvalidDataException, ModelInitializationException, Exception {
		int n = testSet.getNumInstances();
		int k = testSet.getNumLabels();
		int[] labelIndices = testSet.getLabelIndices();
		double[][] predictions = new double[n][k];
		double[][] y = new double[n][k];
		
		for (int i = 0; i < n; ++i) {
			Instance instance = testSet.getDataSet().get(i);
			double[] output = bpmll.makePrediction(instance).getConfidences();
			for (int j = 0; j < k; ++j) {
				double y_hat = output[j];
				double y_truth = instance.value(labelIndices[j]);
				predictions[i][j] = y_hat;
				y[i][j] = y_truth;
			}
		}
		
		return new PredictionResults(predictions, y);
	}
	
}
