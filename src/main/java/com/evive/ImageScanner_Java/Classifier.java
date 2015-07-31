             package com.evive.ImageScanner_Java;

import com.evive.spark.classifier.ClassifierRandomForest;
import com.evive.spark.util.MetricUtil;
import com.evive.spark.util.UtilitySpark;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

public class Classifier {
    public static final Logger LOG = LoggerFactory.getLogger(Classifier.class);
    static Properties properties = Utils.getProperties();

    public static void perdictLabels(RandomForestModel rfModel, String path, String delemitier) {
        final SparkConf sparkConf = new SparkConf().setAppName("LRImageRecognizer").setMaster("local[*]");

        final int lastIndex = path.lastIndexOf("/");
        final StringBuilder fileName = new StringBuilder(path);
        fileName.delete(0, lastIndex);
        final StringBuilder str = new StringBuilder(path);
        str.delete(str.length() - 4, str.length());
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);

        final String[] args1 = { "", path, delemitier };

        final JavaRDD<Vector> testing2 = UtilitySpark.fromCmdArgumentsToVectorJavaRDD(sc, args1);

        final JavaRDD<LabeledPoint> predictedT = testing2.map(f -> {
            return new LabeledPoint(rfModel.predict(f), f);
        });

        final JavaRDD<Integer> labelsPredicted = predictedT.map(f -> (int) f.label());
        // labelsPredicted.coalesce(1).saveAsTextFile(str.toString());
        // final String COMMA_DELIMITER = ",";
        final String NEW_LINE_SEPARATOR = "\n";
        final List<Integer> labels = labelsPredicted.toArray();

        final Path filename = Paths.get(properties.getProperty("predictLabelcsvFolder") + fileName);
        try (BufferedWriter writer = Files.newBufferedWriter(filename)) {
            for (final Integer label : labels) {
                writer.append(String.valueOf(label));
                writer.append(NEW_LINE_SEPARATOR);
            }
        } catch (final IOException e1) {
            e1.printStackTrace();
        }
        LOG.info("Predictedd labels : {}", labelsPredicted.collect());
        for (final LabeledPoint row : predictedT.collect()) {

            LOG.info(" {} ", row);

        }
        sc.stop();
    }

    /**
     * 
     * @param path
     * @param delemiter
     * @return RandomForestModel
     * 
     *         This method takes path of a csv file which is used to train the classifier
     *         
     */
    public static RandomForestModel trainClassifier(String path, String delemiter) {
        final SparkConf sparkConf = new SparkConf().setAppName("LRImageRecognizer").setMaster("local[*]");
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);
        final ClassifierRandomForest rfClassifier = new ClassifierRandomForest();
        final String[] args1 = { "", path, delemiter };
        final JavaRDD<LabeledPoint> training = UtilitySpark.fromCmdArgumentsToLabeledPointJavaRDD(sc, args1);
        training.cache();
        rfClassifier.setNumClasses(11);
        rfClassifier.setNumTrees(100);
        final RandomForestModel rfModel = rfClassifier.trainRandomForestClassifier(training);

        LOG.info("no of training :\n {}", training.count());
        LOG.info("no of training features:\n {}", training.first().features().size());


        LOG.info("Learned classification model:\n {}", rfModel.toString());


        LOG.info("The rf accuracy is {} ", MetricUtil.calculateAccuracy(rfModel, training));

        sc.close();
        return rfModel;
    }
}
