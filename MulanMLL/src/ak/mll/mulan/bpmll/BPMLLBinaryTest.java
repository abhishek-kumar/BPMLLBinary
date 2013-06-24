package ak.mll.mulan.bpmll;

import static org.junit.Assert.fail;

import org.junit.Test;

import ak.mll.mulan.Utility;
import ak.mll.mulan.Utility.PredictionResults;

public class BPMLLBinaryTest {

	@Test
	public void testProcess() {
		
		try {
			/*
			BPMLLBinary.writeToFile(BPMLLBinary.process(
					"/Users/abhishekkr/Documents/data/scene/scene-train.arff",
					"/Users/abhishekkr/Documents/data/scene/scene-test.arff",
					"/Users/abhishekkr/Documents/data/scene/scene.xml", 
					4), "/Users/abhishekkr/Documents/data/scene/predictions.csv");
			
			BPMLLBinary.writeToFile(BPMLLBinary.process(
					"/Users/abhishekkr/Documents/data/yeast/yeast-train.arff",
					"/Users/abhishekkr/Documents/data/yeast/yeast-test.arff",
					"/Users/abhishekkr/Documents/data/yeast/yeast.xml", 
					6), "/Users/abhishekkr/Documents/data/yeast/predictions.csv");
					
			BPMLLBinary.writeToFile(BPMLLBinary.process(
					"/Users/abhishekkr/Documents/data/emotions/emotions-train.arff",
					"/Users/abhishekkr/Documents/data/emotions/emotions-test.arff",
					"/Users/abhishekkr/Documents/data/emotions/emotions.xml", 
					4), "/Users/abhishekkr/Documents/data/emotions/predictions.csv");
				
			BPMLLBinary.writeToFile(BPMLLBinary.process(
					"/Users/abhishekkr/Documents/data/enron/enron-train.arff",
					"/Users/abhishekkr/Documents/data/enron/enron-test.arff",
					"/Users/abhishekkr/Documents/data/enron/enron.xml", 
					10), "/Users/abhishekkr/Documents/data/enron/predictions.csv");
					
			BPMLLBinary.writeToFile(BPMLLBinary.process(
					"/Users/abhishekkr/Documents/data/moviegenre/moviegenre.arff",
					"/Users/abhishekkr/Documents/data/moviegenre/moviegenre.arff",
					"/Users/abhishekkr/Documents/data/moviegenre/moviegenre.xml", 
					10), "/Users/abhishekkr/Documents/data/moviegenre/predictions.csv");
		
			doProcess("/Users/abhishekkr/Documents/data/yeast/yeast-train.arff", 
					"/Users/abhishekkr/Documents/data/yeast/yeast-test.arff", 35, 
					"/Users/abhishekkr/Documents/data/yeast/yeast.xml", "/Users/abhishekkr/Documents/data/yeast/predictions.csv", "/Users/abhishekkr/Documents/data/yeast/groundtruth.csv");
					*/
		} catch (Exception ex) {
			fail("MLL prediction failed. Exception: \n" +
					ex.toString());
		}
		
	}
	
	@Test
	public void testProcessCV() {
		String trainingFile, testFile, xmlFile, datasetDirectory;
		try {
			datasetDirectory = "/Users/abhishekkr/Documents/data/";
			trainingFile = datasetDirectory + "yeast/yeast-train.arff";
			testFile = datasetDirectory + "yeast/yeast-test.arff";
			xmlFile = datasetDirectory + "yeast/yeast.xml";
			PredictionResults results = BPMLLBinary.tuneParamsAndCrossValidate(
					Utility.mergeTwoDatasets(trainingFile, testFile, xmlFile).getDataSet(), 
					xmlFile, 10, 40);
			results.writeToDirectory(datasetDirectory + "yeast/");
			
			//doCV("/Users/abhishekkr/Documents/data/scene/scene-train.arff", 4, 
			//		"/Users/abhishekkr/Documents/data/scene/scene.xml", 
			//		"/Users/abhishekkr/Documents/data/scene/predictions.csv",
			//		"/Users/abhishekkr/Documents/data/scene/groundtruth.csv");
			
			//doCV(, 
			//		"/Users/abhishekkr/Documents/data/yeast/yeast-test.arff", 30, 
			//		"/Users/abhishekkr/Documents/data/yeast/yeast.xml", 
			//		"/Users/abhishekkr/Documents/data/yeast/predictions.csv",
			//		"/Users/abhishekkr/Documents/data/yeast/groundtruth.csv");
			
			//doCV("/Users/abhishekkr/Documents/data/emotions/emotions-train.arff", 4, 
			//		"/Users/abhishekkr/Documents/data/emotions/emotions.xml", 
			//		"/Users/abhishekkr/Documents/data/emotions/predictions.csv",
			//		"/Users/abhishekkr/Documents/data/emotions/groundtruth.csv");
			
		} catch (Exception ex) {
			fail("MLL prediction failed. Exception: \n" +
					ex.toString());
		}
		
	}
}
