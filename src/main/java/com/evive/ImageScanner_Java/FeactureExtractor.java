package com.evive.ImageScanner_Java;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * 
 * @author abhishek
 *
 */
public class FeactureExtractor {
    private static final Logger LOG = LoggerFactory.getLogger(FeactureExtractor.class);

    /**
     * 
     * @param Mat image
     * @param List<PointList> squares
     * @param String filename
     * 
     *        Now pass the image and the List of coordinates to collectHOG function which extracts the roi in the image
     *        based on the coordinates and extracts HOG features for that roi and writes the feature vector into a csv
     *        file
     * @throws IOException
     */
    public static void collectHOG(Mat image, List<PointList> squares, String filename) throws IOException {


        if (image.empty() || squares.isEmpty()) {
            LOG.error("Error in the parameters");

        } else {
            LOG.info("Computing HOG");
            //Sort the squares based on y- axis as the form is fixed we will be able to sort 
            // the rectangles according to the line in which it is present.
            final List<PointList> sqr = ProcessForm.sortPtsAccordY(squares);
            
            //Sort the above List of PointList(squares) according to their position in a line( refer to the form)
            List<PointList> sqr1 = ProcessForm.sortSquaresInLine(sqr);
            
            //Remove all the squares which contain non numerical values.
            sqr1 = ProcessForm.removeNonNumeric(sqr1);
            
            
            final String COMMA_DELIMITER = ",";
            final String NEW_LINE_SEPARATOR = "\n";
            LOG.info("Writing to csv");
            final Path path = Paths.get(filename);
            try (BufferedWriter writer = Files.newBufferedWriter(path)) {
                for (int i = 0; i < sqr1.size(); i++) {
                    // LOG.info("Squares : {}",sqr1.get(i));
                    
                    //Extract a part of the image(Form) i.e the square using the Points.
                    final MatOfPoint mat = new MatOfPoint();
                    mat.fromList(sqr1.get(i).getRect());
                    final Rect rect = Imgproc.boundingRect(mat);
                    final Mat roi = new Mat(image, rect);
                    //String name = i + ".jpg";
                    //Highgui.imwrite(name,roi);

                    //Extract the HOG feature vector of the ROI. 
                    final MatOfFloat descriptor = FeactureExtractor.getHOGFeatures(roi);
                    final List<Float> featureVector = descriptor.toList();
                    
                    //Write the values to a file(csv)
                    for (final Float feaValue : featureVector) {
                        writer.append(String.valueOf(feaValue));
                        writer.append(COMMA_DELIMITER);
                    }
                    writer.append(NEW_LINE_SEPARATOR);
                }

            } catch (final IOException e2) {
                throw e2;
            }
            
        }
    }

    /**
     * 
     * @param Mat image
     * @return MatOfFloat
     */
    public static MatOfFloat getHOGFeatures(Mat image) {

        //Pre process the image i.e change the image to gray scale and then smoothen it using Gaussian blur then
        //apply adaptive threshold to convert it into black and white image.
        final Mat processImage = preProcessImage(image);
        
        //Removes the lines around the digit in the per-processed image.
        final Mat final_image = removeBoundries(processImage);
        
        Imgproc.resize(final_image, final_image, new Size(64, 64));
        final HOGDescriptor hog =
                new HOGDescriptor(new Size(32, 32), new Size(32, 32), new Size(16, 16), new Size(16, 16), 9);
        final MatOfPoint locations = new MatOfPoint();
        final MatOfFloat descriptors = new MatOfFloat();
        hog.compute(final_image, descriptors, new Size(32, 32), new Size(0, 0), locations);
        LOG.info(" Descriptor : {} ", descriptors.toList());
        return descriptors;
    }

    /**
     * 
     * @param Mat image
     * @return Mat 
     */
    public static Mat preProcessImage(Mat image) {
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(image, image, new Size(9, 9), 0, 0);
        Imgproc.adaptiveThreshold(image, image, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11,
                2);
        return image;
    }

    /**
     * 
     * @param Mat image
     * @return Mat 
     */
    public static Mat removeBoundries(Mat image) {
        final Mat new_image = image;
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                if (i > 10 && j < image.cols() - 5 && i < image.rows() - 5 && j > 10) {
                    if (image.get(i, j)[0] < 255) {
                        new_image.put(i, j, 0);
                    } else {
                        new_image.put(i, j, 255);
                    }
                } else {
                    new_image.put(i, j, 255);
                }
            }
        }

        return new_image;
    }

}
