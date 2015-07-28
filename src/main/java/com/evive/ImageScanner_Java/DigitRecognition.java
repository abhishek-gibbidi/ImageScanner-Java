package com.evive.ImageScanner_Java;

import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class DigitRecognition {
    public static final Logger LOG = LoggerFactory.getLogger(DigitRecognition.class);
    static Properties properties = Utils.getProperties();

    /**
     * 
     * @param args
     * @throws IOException 
     */
    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // gr.assignLabels();
        for (int i = 0; i < 70; i++) {

            final StringBuilder str = new StringBuilder(String.valueOf(properties.get("inputFormFolder")));
            str.append("form-").append(i).append(String.valueOf(properties.get("IMAGE_EXTENSTION")));
            final StringBuilder imageName = new StringBuilder();
            imageName.append("form-").append(i).append(".png");
            LOG.info(
                    "***********************************************************IMAGE : {} ********************************************",
                    str);
            final StringBuilder csvName = new StringBuilder(String.valueOf(properties.get("csvFolder")));
            csvName.append("form-").append(i).append(".csv");
            List<PointList> squares = new ArrayList<>();
            LOG.info("Path : {}", str);


            // Read Image(form)
            final Mat image = Highgui.imread(str.toString());

            // Resize the image to 1000*1000 so that all the uploaded images
            // will be of same dimensions.
            Imgproc.resize(image, image, new Size(1000, 1000));

            // Pass the image to findSquares function which finds all the required squares in
            // the image and gives out a List of PointList(see PointList class) which contains the
            // coordinates of all the squares.
            squares = GetRectangles.findSquareV2(image, squares);
            LOG.info("Squares Size : {}", squares.size());

            
            // Now pass the image and the List of coordinates to collectHOG function
            // which extracts the roi in the image based on the coordinates and extracts
            // HOG features for that roi and writes the feature vector into a csv file
            FeactureExtractor.collectHOG(image, squares, csvName.toString());

        }
        final File dir = new File(properties.getProperty("predictLabelcsvFolder"));
        final boolean successful = dir.mkdir();
        if (!successful) {
            LOG.info("Unable to create dir {} ", properties.getProperty("predictLabelcsvFolder"));
        }

        // TrainClassifier takes a csv file and the delimiter and trains the RandomForestModel using the csv
        // here each row contains a label followed by a feature vector.
        final RandomForestModel rfModel =
                Classifier.trainClassifier("/home/abhishek/Desktop/HWR/DigitRecognition/HOGfea.csv", ",");

        Classifier.perdictLabels(rfModel, "/home/abhishek/Desktop/HWR/DigitRecognition/csvFolder/form-2.csv", ",");
        GetRectangles.assignLabels("/home/abhishek/Desktop/HWR/DigitRecognition/csvFolder/predictLabelsCSV/form-2.csv");
    }

}
