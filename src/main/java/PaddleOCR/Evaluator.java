package PaddleOCR;

// TODO: remove all camel cases
// TODO: write author before every class

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

import helpers.ImageHandler;
import helpers.Converters;
import helpers.DRAWMODE;


/* ***   FINAL CSV   ***

Information about picture
 - imageName
 - xResolution, yResolution

Information about time
 - msDet
 - msRec
 - msTot

DETECTIONS:
Information about how many shapes were detected
 - tp, fn, fp

Information about how good the detected shapes are overlapping with the ground truth
 - iouAverage

RECOGNITIONS:
jaccard trigram distance
jaccard distance (on characters), i.e. iou
 */

/**
 * Creates a csv file containing:
 *
 *  image_name,     name of the image
 *  x_res,          width of the image
 *  y_res,          height of the image
 *  ms_det,         time it took to do text detection
 *  ms_rec,         time it took to recognize the text in all textboxes
 *  ms_tot,         time it took to do text detection and text recognition
 *  tp,             true positives: textboxes that were detected correctly
 *  fn,             false negatives: textboxes in ground truth whose text was not detected
 *  fp,             false positives: textboxes detected where there is no text
 *  iou_avg,        average iou of detected textboxes: sum of all tp iou's divided by number of tp
 *  jaccard_distance_recognition,
 *                  intersection over union
 *  jaccard_trigram_distance_recognition,
 *                  sum of jaccard trigram distances between words found and words in ground truth
 */
public class Evaluator {

    private class DetectionEvaluationResult {
        double avgIOU;      // intersection over union
        Set<Textbox> tp, fn, fp;  // true positives, false negatives, false positives

        DetectionEvaluationResult(double avgIOU, Set<Textbox> tp, Set<Textbox> fn, Set<Textbox> fp) {
            this.avgIOU = avgIOU;
            this.tp = tp;
            this.fn = fn;
            this.fp = fp;
        }
    }

    private class RecognitionEvaluationResult {
        double iou, jaccard_trigram_distance;

        RecognitionEvaluationResult(double iou, double jaccardTrigramDistance) {
            this.iou = iou;
            this.jaccard_trigram_distance = jaccardTrigramDistance; // würg
        }
    }

    public void evaluate_incidental_scene_text() {
        // These are user inputs. TODO: Implement argument builder
        final String file_name_results = "results.csv";
        final String path_to_images = "test_images/IncidentalSceneText/test/";
        final String path_to_ground_truth = "test_images/IncidentalSceneText/test/gt/";
        final String path_to_output = "evaluations/";
        final boolean save_output_images = true;

        File img_folder = new File(path_to_images);
        String[] imageNames = img_folder.list((dir, name) -> name.endsWith(".jpg"));
        assert imageNames != null;
        // TODO: find a solution for img sorting
        // Problem: this sorts images lexicographically, i.e.
        // img_1, img_10, img_100, img_101, img_102, ...
        //Arrays.sort(imageNames);

        InferenceModel inferenceModel = new InferenceModel();

        try (Writer w = new FileWriter(file_name_results)) {
            //try (Reader r = new FileReader(ground_truth))
            BufferedWriter csvWriter = new BufferedWriter(w);
            csvWriter.append("image_name,x_res,y_res,ms_det,ms_rec,ms_tot,tp,fn,fp,iou_avg,jaccard_distance_recognition,jaccard_trigram_distance_recognition");
            csvWriter.append(System.lineSeparator());

            // Define variables
            long start_det, end_det;  // detection
            long start_rec, end_rec;  // recognition
            long ms_det, ms_rec, ms_tot;  // milliseconds

            DetectionEvaluationResult detectionEvaluationResult;
            RecognitionEvaluationResult recognitionEvaluationResult;

            Image img;

            HashMap<String, List<Textbox>> groundTruthTextboxes = getGroundTruthTextboxes(path_to_ground_truth);
            DetectedObjects detectedBoxes;

            for (String imageName : imageNames) {
                // 1. Load image
                img = inferenceModel.loadImage(path_to_images, imageName);

                // 2. Text detection
                start_det = System.currentTimeMillis();
                detectedBoxes = inferenceModel.detection(img);
                end_det = System.currentTimeMillis();

                // 3. Text recognition
                start_rec = System.currentTimeMillis();
                List<String> recognizedText = inferenceModel.recognition(img, detectedBoxes);
                end_rec = System.currentTimeMillis();

                // 4. Evaluate detections
                System.out.println("Evaluating detections of img: " + imageName + " ...");
                String key = imageName.split("\\.", 2)[0];
                List<Textbox> gtTextboxList = groundTruthTextboxes.get(key);
                detectionEvaluationResult = evaluateDetections(img, detectedBoxes, gtTextboxList);

                // 5. Evaluate recognitions
                recognitionEvaluationResult = evaluateRecognitions(recognizedText); // TODO: add ground truth

                // 6. Calculate runtimes
                ms_tot = end_rec - start_det;
                ms_det = end_det - start_det;
                ms_rec = end_rec - start_rec;

                // 7. Write data to file
                String line = format_csv_line(imageName, img.getWidth(), img.getHeight(),
                        ms_det, ms_rec, ms_tot,
                        detectionEvaluationResult.tp.size(),
                        detectionEvaluationResult.fn.size(),
                        detectionEvaluationResult.fp.size(),
                        detectionEvaluationResult.avgIOU,
                        recognitionEvaluationResult.iou,
                        recognitionEvaluationResult.jaccard_trigram_distance);
                csvWriter.write(line);
                System.out.println(line);

                // 8. Draw textboxes on the picture
                BufferedImage bufferedImage = ImageHandler.loadImage(path_to_images + imageName);
                if (save_output_images) {

                    // Draw all boxes
                    drawBoxes(
                            bufferedImage,
                            gtTextboxList,
                            DRAWMODE.gt);
                    drawBoxes(
                            bufferedImage,
                            List.copyOf(detectionEvaluationResult.fn),
                            DRAWMODE.fn);
                    drawBoxes(
                            bufferedImage,
                            List.copyOf(detectionEvaluationResult.fp),
                            DRAWMODE.fp);
                    drawBoxes(
                            bufferedImage,
                            List.copyOf(detectionEvaluationResult.tp),
                            DRAWMODE.tp);

                    // Draw a color legend
                    drawColorLegend(bufferedImage);

                    // Save image
                    String name = imageName.split("\\.")[0];  // e.g.: img_1
                    String type = imageName.split("\\.")[1];  // e.g.: jpg

                    ImageHandler.saveImage(bufferedImage, path_to_output, name + "_detection_result", type);
                }

            }
            csvWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private void drawBoxes(BufferedImage img, List<Textbox> textboxes, DRAWMODE mode) { // e.g.: "evaluations/", "img_2.png"

        Color color = DRAWMODE.getColor(mode);

        for (Textbox textbox : textboxes) {
            ImageHandler.drawRectangle(img,
                    textbox.xMin,
                    textbox.xMax,
                    textbox.yMin,
                    textbox.yMax,
                    color);
        }
    }

    private void drawColorLegend(BufferedImage img) {
        int linespace = 2;
        int textheight = 10; // THIS DOES NOT CHANGE TEXT SIZE


        List<DRAWMODE> order = new ArrayList<DRAWMODE>(
                Arrays.asList(
                        DRAWMODE.tp,
                        DRAWMODE.fp,
                        DRAWMODE.gt,
                        DRAWMODE.fn));
        Collections.reverse(order);

        int y = img.getHeight() - 10;
        for (DRAWMODE mode : order) {
            ImageHandler.drawText(img, DRAWMODE.getDescription(mode), 10, y, DRAWMODE.getColor(mode));
            y -= (textheight + linespace);
        }
    }

    private RecognitionEvaluationResult evaluateRecognitions(List<String> recognizedText) {
        return new RecognitionEvaluationResult(1.0d, 1.0d);
    }

    private HashMap<String, List<Textbox>> getGroundTruthTextboxes(String path_to_ground_truth) {
        // pic: img_1.jpg
        // gt : gt_img_1.txt
        // key: img_1
        HashMap<String, List<Textbox>> hashmap = new HashMap<>();
        File gt_folder = new File(path_to_ground_truth);
        String[] files = gt_folder.list((dir, name) -> name.endsWith(".txt"));
        for (String fileName : files) {
            List<Textbox> list = new ArrayList<>();
            File file = new File(path_to_ground_truth + fileName);
            String imgName = (fileName.split("_", 2)[1]).split("\\.", 2)[0];

            final Scanner s;
            try {
                s = new Scanner(file);
                while(s.hasNextLine()) {
                    final String line = s.nextLine();
                    list.add(Textbox.fromIncidentalSceneTextTXT(line));
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            hashmap.put(imgName, list);
        }
        return hashmap;
    }

    private DetectionEvaluationResult evaluateDetections(Image img, DetectedObjects detectedObjects, List<Textbox> groundTruthTextboxes) {
        int size = detectedObjects.getNumberOfObjects();

        Set<Textbox> tp = new HashSet<>();
        Set<Textbox> fn = new HashSet<>();
        Set<Textbox> fp = new HashSet<>();

        double avgIOU = 0;
        Set<Textbox> foundBoxes = new HashSet<>();

        List<Textbox> detectedTextboxes = Converters.DetectedObjects2ListOfTextboxes(detectedObjects, img.getWidth(), img.getHeight());
        int i = 0;
        for (Textbox gtBox : groundTruthTextboxes) {
            double maxIoU = 0;
            Textbox fittingTextbox = null;


            for (Textbox detectedTextbox : detectedTextboxes) {
                double iou = gtBox.getIoU(detectedTextbox);
                if (iou > maxIoU) {
                    maxIoU = iou;
                    fittingTextbox = detectedTextbox;
                }
            }

            if (fittingTextbox != null) {
                tp.add(fittingTextbox);
                foundBoxes.add(gtBox);
            } else {
                fn.add(gtBox);
            }


            avgIOU += maxIoU;

        }

        if (tp.isEmpty()) {
            avgIOU = 0;
        } else {
            avgIOU /= tp.size();
        }
        detectedTextboxes.removeAll(tp);
        fp = Set.copyOf(detectedTextboxes);


        //System.out.println("average IOU: " + avgIOU);
        //System.out.println("tp: " + tp.size());
        //System.out.println("fn: " + fn.size());
        //System.out.println("fp: " + fp.size());

        //System.out.println("=================");

        return new DetectionEvaluationResult(avgIOU, tp, fn, fp);
    }

    private String format_csv_line(String image_name,
                                 int x_res,
                                 int y_res,
                                 long ms_det,
                                 long ms_rec,
                                 long ms_tot,
                                 int tp,
                                 int fn,
                                 int fp,
                                 double iou_avg,
                                 double jaccard_distance_recognition,
                                 double jaccard_trigram_distance_recognition)
            throws IOException {

        StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append(image_name);
        stringBuilder.append(",");

        stringBuilder.append(Integer.toString(x_res));
        stringBuilder.append(",");

        stringBuilder.append(Integer.toString(y_res));
        stringBuilder.append(",");

        stringBuilder.append(Long.toString(ms_det));
        stringBuilder.append(",");

        stringBuilder.append(Long.toString(ms_rec));
        stringBuilder.append(",");

        stringBuilder.append(Long.toString(ms_tot));
        stringBuilder.append(",");

        stringBuilder.append(Integer.toString(tp));
        stringBuilder.append(",");

        stringBuilder.append(Integer.toString(fn));
        stringBuilder.append(",");

        stringBuilder.append(Integer.toString(fp));
        stringBuilder.append(",");

        stringBuilder.append(Double.toString(iou_avg));
        stringBuilder.append(",");

        stringBuilder.append(Double.toString(jaccard_distance_recognition));
        stringBuilder.append(",");

        stringBuilder.append(Double.toString(jaccard_trigram_distance_recognition));


        stringBuilder.append(System.lineSeparator());

        return stringBuilder.toString();
    }

    public static void main(String[] args) {
        Evaluator evaluator = new Evaluator();
        evaluator.evaluate_incidental_scene_text();


    }

}


