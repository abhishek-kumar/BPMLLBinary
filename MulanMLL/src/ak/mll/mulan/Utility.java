package ak.mll.mulan;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

import com.sun.tools.javac.util.Pair;

/**
 * Utility functions to be used by the MLL models.
 * 
 * @author Abhishek Kumar (abhishek.kumar.ak@gmail.com)
 *
 */
public class Utility {

	/**
	 * Return the log likelihood of a given instance, P(Y = y | model)
	 * @param y the true value of label Y.
	 * @param yp the probability P(Y = 1) based on our model's prediction.
	 * @return the log-likelihood log P(Y = y | model)
	 */
	public static double logLikelihood(double y, double yp) {
		// prevent overflow errors
		double epsilon = 1e-10;
		double yp1 = yp + epsilon;
		double yp2 = yp - epsilon;
		
		// Log likelihood
		return y*Math.log(yp1)+(1-y)*Math.log(1-yp2);
	}
	
	/**
	 * Writes string data to a file. If the file already exists, 
	 * it is deleted first.
	 * @param data the data to write.
	 * @param outputFileName the full path to the output file.
	 * @throws IOException
	 */
	public static void writeToFile(String data, String outputFileName) 
			throws IOException {
		File outputFile = new File(outputFileName);
		if (outputFile.exists()) {
			outputFile.delete();
		}
		FileOutputStream outputStream = new FileOutputStream(new File(outputFileName));
		outputStream.write(data.getBytes());
		outputStream.close();
	}
	
	/**
	 * Converts a set of label values to CSV format.
	 * @param values a set of label values to convert. 
	 * Dimensions n by k where n = number instances, and k = number of labels. 
	 * @return a {@code String} representation of comma separated values of 
	 * {@code values}.
	 */
	public static String convertToCsv(double[][] values) {
		StringBuilder sb = new StringBuilder();
		int n = values.length;
		if (n == 0) {
			return "";
		}
		int k = values[0].length;
		
		// Write as comma separated values
		for (int i = 0; i < n; ++i) {
			String comma = "";
			for (int j = 0; j < k; ++j) {
				sb.append(comma).append(values[i][j]);
				comma = ",";
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * Reads two files and returns a dataset comprising instances from both. This
	 * is helpful while combining a training set and test set for doing cross validation.
	 * <p> Warning: The dimensions of both datasets must match. 
	 * @param dataset1 File path to the first dataset. Must not be a wrong path or null.
	 * @param dataset2 File path to the second dataset. Must not be a wrong path or null.
	 * @param xmlFile File path to the XML File that describes the labels.
	 * @return dataset with all instances from both files.
	 * @throws InvalidDataFormatException 
	 */
	public static MultiLabelInstances mergeTwoDatasets(
			String dataset1, String dataset2, String xmlFile)
					throws InvalidDataFormatException {
		MultiLabelInstances data1 = new MultiLabelInstances(dataset1, xmlFile);
		MultiLabelInstances data2 = new MultiLabelInstances(dataset2, xmlFile);
		Instances data = data1.getDataSet();
		for (Instance i : data2.getDataSet()) {
			data.add(i);
		}
		
		return new MultiLabelInstances(data, xmlFile);
	}
	
	/**
	 * Placeholder for prediction results. Pair {first, second} where
	 * first = predictions, and second = ground truth.
	 * 
	 * @author Abhishek Kumar (abhishek.kumar.ak@gmail.com)
	 */
	public static class PredictionResults extends Pair<double[][], double[][]> {
		// n: number of instances; k = number of labels.
		int n, k;
		
		public PredictionResults(double[][] predictions, double[][] groundtruth) {
			super(predictions, groundtruth);
			n = predictions.length;
			k = predictions[0].length;
		}
		
		public static PredictionResults add(PredictionResults oldPredictions, PredictionResults newPredictions) {
			if (oldPredictions == null) {
				return newPredictions;
			}
			int n = oldPredictions.fst.length + newPredictions.fst.length;
			int k = oldPredictions.fst[0].length;
			double[][] predictions = new double[n][k];
			double[][] y = new double[n][k];
			
			for (int i = 0; i < oldPredictions.fst.length; ++i) {
				for (int j = 0; j < k; ++j){
					predictions[i][j] = oldPredictions.fst[i][j];
					y[i][j] = oldPredictions.snd[i][j];
				}
			}
			
			for (int i = oldPredictions.fst.length; i < n; ++i) {
				for (int j = 0; j < k; ++j){
					predictions[i][j] = newPredictions.fst[i - oldPredictions.fst.length][j];
					y[i][j] = newPredictions.snd[i - oldPredictions.fst.length][j];
				}
			}
			return new PredictionResults(predictions, y);
			
		}
		
		public double getLogLikelihood() {
			double LL = 0.0;
			
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < k; ++j) {
					LL += logLikelihood(super.snd[i][j], super.fst[i][j]);
				}
			}
			return LL;
		}
		
		public void writeToDirectory(String directory) throws IOException {
			if (directory == null) {
				throw new IOException("Empty directory provide.");
			}
			String predictionFile = directory + File.separator + "predictions.csv";
			String groundTruthFile = directory + File.separator + "groundtruth.csv";
			writeToFile(convertToCsv(super.fst), predictionFile);
			writeToFile(convertToCsv(super.snd), groundTruthFile);
		}
	}
}
