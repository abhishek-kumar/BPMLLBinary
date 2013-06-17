import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.BPMLL;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Utils;

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

	/**
	 * Main entry point. Trains a model on the training set, and makes predictions for the
	 * test set.
	 * @param args 4 arguments expected. 'train', 'test', 'xml' and 'output' for the
	 * training file, test file, xml header file, and output file locations respectively.
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		logger.setLevel(Level.INFO);
		
		// Read flags
		String trainingFile = Utils.getOption("train", args);
		String testFile = Utils.getOption("test", args);
		String xmlFile= Utils.getOption("xml", args);
		String outputFile = Utils.getOption("output", args);
		
		// Train the model, make predictions and write results to file
		String predictions = process(trainingFile, testFile, xmlFile, null);
		writeToFile(predictions, outputFile);
	}
	
	/**
	 * Use the MULAN neural network model with one hidden layer to make predictions
	 * for a multilabel data set.
	 * @param trainingFile the full file path of the training file in ARFF format.
	 * @param testFile the full file path of the test file in ARFF format.
	 * @param xmlFile the full file path of the xml file describing the labels.
	 * @param numHiddenUnits number of hidden units to use in the model.
	 * @return Predicted results in CSV, serialized to a {@String}. 
	 * @throws Exception
	 */
	private static String process(String trainingFile, String testFile, 
			String xmlFile, Integer numHiddenUnits) throws Exception {
		logger.info("Processing dataset: " + trainingFile + 
				"\n with " + numHiddenUnits + " hidden units.");
		
		// Read and initialize data sets
		MultiLabelInstances dataset = new MultiLabelInstances(trainingFile, xmlFile);
		MultiLabelInstances testset = new MultiLabelInstances(testFile, xmlFile);
		StringBuilder sb = new StringBuilder();
		
		// Build and train the model
		BPMLL bpmll = new BPMLL();
		bpmll.setTrainingEpochs(1000);
		if (numHiddenUnits != null) {
			bpmll.setHiddenLayers(new int[]{numHiddenUnits});
		}
		bpmll.build(dataset);
				
		// Make predictions for the test set
		for (Instance i : testset.getDataSet()) {
			int count = 1;
			MultiLabelOutput o = bpmll.makePrediction(i);
			double[] output = o.getConfidences();
			String comma = "";
			for (double v : output) {
				sb.append(comma).append(v);
				comma = ",";
			}
			sb.append("\n");
			logger.fine("Instance #" + count++ + "\tPrediction: " + o.toString());
		}
		return sb.toString();
	}
	
	/**
	 * Writes string data to a file. If the file already exists, 
	 * it is deleted first.
	 * @param data the data to write.
	 * @param outputFileName the full path to the output file.
	 * @throws IOException
	 */
	private static void writeToFile(String data, String outputFileName) 
			throws IOException {
		File outputFile = new File(outputFileName);
		if (outputFile.exists()) {
			outputFile.delete();
		}
		FileOutputStream outputStream = new FileOutputStream(new File(outputFileName));
		outputStream.write(data.getBytes());
		outputStream.close();
	}

}
