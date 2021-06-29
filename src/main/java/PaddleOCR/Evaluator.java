package PaddleOCR;

// TODO: remove all camel cases
// TODO: write author before every class

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

import helpers.ImageHandler;
import helpers.Converters;

/**
 * Creates a csv file containing (currently the first option):
 *
 *  image_name,     name of the image
 *  ms_tot,         time it took to do text detection and text recognition
 *  ms_det,         time it took to do text detection
 *  iou_det,        intersection over union of textboxes
 *  max_stretch,    maximal value of length and width
 *  num_subimages,  number of textboxes found
 *  ms_rec,         time it took to recognize the text in all textboxes
 *  iou_rec,        intersection over union of letters
 *  jaccard_trigram_distance    sum of jaccard trigram distances between words found and words in ground truth
 *
 * OR
 * // TODO: write this
 *  image_name,     name of the image
 *
 *  ms_det,
 *  ms_rec,
 *  ms_tot,
 *
 *  iou,
 *  tp,
 *  fn,
 *  fp,
 *
 *  iou,
 *  jaccardTrigramDistance
 */
public class Evaluator {

    private class DetectionEvaluationResult {
        double iou;      // intersection over union
        int tp, fn, fp;  // true positives, false negatives, false positives

        DetectionEvaluationResult(double iou, int tp, int fn, int fp) {
            this.iou = iou;
            this.tp = tp;
            this.fn = fn;
            this.fp = fp;
        }
    }

    private class RecognitionEvaluationResult {
        double iou, jaccardTrigramDistance;

        RecognitionEvaluationResult(double iou, double jaccardTrigramDistance) {
            this.iou = iou;
            this.jaccardTrigramDistance = jaccardTrigramDistance;
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
            csvWriter.append("image_name,ms_tot,ms_det,iou_det,max_stretch,num_subimages,ms_rec,iou_rec,jaccard_trigram_distance");
            csvWriter.append(System.lineSeparator());

            // Define variables
            long start_det, end_det;  // detection
            long start_rec, end_rec;  // recognition
            long ms_det, ms_rec, ms_tot;  // milliseconds

            DetectionEvaluationResult detectionEvaluationResult;
            RecognitionEvaluationResult recognitionEvaluationResult;

            Image img;
            int max_stretch;
            int num_subimages;

            //Array of textbox lists

            HashMap<String, List<Textbox>> groundTruthTextboxes = getGroundTruthTextboxes(path_to_ground_truth);
            DetectedObjects detectedBoxes;

            for (String imageName : imageNames) {
                // 1. Load image
                img = inferenceModel.loadImage(path_to_images, imageName);
                max_stretch = Math.max(img.getHeight(), img.getWidth());

                // 2. Text detection
                start_det = System.currentTimeMillis();
                detectedBoxes = inferenceModel.detection(img);
                end_det = System.currentTimeMillis();

                // 3. Text recognition
                start_rec = System.currentTimeMillis();
                List<String> recognizedText = inferenceModel.recognition(img, detectedBoxes);
                end_rec = System.currentTimeMillis();

                // 4. Evaluate detections
                System.out.println("Getting iou_det of img: " + imageName);
                String key = imageName.split("\\.", 2)[0];
                List<Textbox> gtTextboxList = groundTruthTextboxes.get(key);
                detectionEvaluationResult = evaluateDetections(img, detectedBoxes, gtTextboxList);

                // 5. Evaluate recognitions
                recognitionEvaluationResult = evaluateRecognitions(recognizedText); // TODO: add ground truth
                num_subimages = recognizedText.size();

                // 6. Calculate runtimes
                ms_tot = end_rec - start_det;
                ms_det = end_det - start_det;
                ms_rec = end_rec - start_rec;

                // 7. Write data to file
                // TODO: Use either option one or option two
                append_line_option_one(csvWriter, imageName, ms_tot, ms_det,
                        detectionEvaluationResult.iou, max_stretch, num_subimages, ms_rec, recognitionEvaluationResult.iou, recognitionEvaluationResult.jaccardTrigramDistance);

                // TODO: draw textboxes
                // 8. Draw textboxes on the picture
                List<Textbox> textboxesList = Converters.DetectedObjects2ListOfTextboxes(detectedBoxes, img.getWidth(), img.getHeight());

                BufferedImage bufferedImage = ImageHandler.loadImage(path_to_images + imageName);
                if (save_output_images) {
                    drawBoxes(
                            bufferedImage,
                            textboxesList,
                            DRAWMODE.predictionExtended,
                            path_to_output,
                            imageName);
                    drawBoxes(
                            bufferedImage,
                            gtTextboxList,
                            DRAWMODE.gt,
                            path_to_output,
                            imageName);

                }

            }
            csvWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private enum DRAWMODE {
        gt,
        predictionExtended,
        prediction,
    }

    // CARE: THIS WILL CHANGE THE BUFFERED IMAGE img!
    private void drawBoxes(BufferedImage img, List<Textbox> textboxes, DRAWMODE mode,
                           String outputImageFolder, String imageName) { // e.g.: "evaluations/", "img_2.png"
        assert imageName.contains("\\.");

        Color color = Color.WHITE; // default
        switch (mode) {
            case gt:
                color = Color.BLACK;  break;

            case predictionExtended:
                color = Color.GREEN;   break;

            case prediction:
                color = Color.CYAN;   break;
        }

        for (Textbox textbox : textboxes) {
            ImageHandler.drawRectangle(img,
                    textbox.xMin,
                    textbox.xMax,
                    textbox.yMin,
                    textbox.yMax,
                    color);
        }

        String name = imageName.split("\\.")[0];
        String type = imageName.split("\\.")[1];
        ImageHandler.saveImage(img, outputImageFolder, name, type);
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
        System.out.println("Start get_iou_det");
        System.out.println("ground truth: " + groundTruthTextboxes);
        int size = detectedObjects.getNumberOfObjects();
        System.out.println("Detected textboxes: " + size);
        System.out.println("Textboxes in ground truth: " + groundTruthTextboxes.size());

        int tp = 0, fn = 0, fp = 0;
        double finalIoU = 0;
        Set<Textbox> foundBoxes = new HashSet<>();

        List<Textbox> textboxes = Converters.DetectedObjects2ListOfTextboxes(detectedObjects, img.getWidth(), img.getHeight());
        int i = 0;
        for (Textbox detectedTextbox : textboxes) {
            System.out.println("[" + (i++) + "] " + detectedTextbox);

            double maxIoU = 0;
            Textbox fittingTextbox = null;
            for (Textbox gtBox : groundTruthTextboxes) {
                double iou = detectedTextbox.getIoU(gtBox);
                if (iou > maxIoU) {
                    maxIoU = iou;
                    fittingTextbox = gtBox;
                }
            }

            if (fittingTextbox != null) {
                tp++;
                foundBoxes.add(fittingTextbox);
            } else {
                fp++;
            }


            finalIoU += maxIoU;

            System.out.println("Fitting ground truth: " + fittingTextbox);
            System.out.println("with IoU: " + maxIoU);

        }

        if (tp == 0) {
            finalIoU = 0;
        } else {
            finalIoU /= tp;
        }
        fn = groundTruthTextboxes.size() - foundBoxes.size();

        System.out.println("finalIoU: " + finalIoU);
        System.out.println("tp: " + tp);
        System.out.println("fn: " + fn);
        System.out.println("fp: " + fp);

        System.out.println("=================");

        return new DetectionEvaluationResult(finalIoU, tp, fn, fp);
    }

    private void append_line_option_one(Writer writer,
                             String image_name,
                             long ms_tot,
                             long ms_det,
                             double iou_det,
                             int max_stretch,
                             int num_subimages,
                             long ms_rec,
                             double iou_rec,
                             double jaccard_trigram_distance) throws IOException {

        writer.append(image_name);
        writer.append(",");

        writer.append(Long.toString(ms_tot));
        writer.append(",");

        writer.append(Long.toString(ms_det));
        writer.append(",");

        writer.append(Double.toString(iou_det));
        writer.append(",");

        writer.append(Integer.toString(max_stretch));
        writer.append(",");

        writer.append(Integer.toString(num_subimages));
        writer.append(",");

        writer.append(Long.toString(ms_rec));
        writer.append(",");

        writer.append(Double.toString(iou_rec));
        writer.append(",");

        writer.append(Double.toString(jaccard_trigram_distance));


        writer.append(System.lineSeparator());
    }

    public static void main(String[] args) {
        Evaluator evaluator = new Evaluator();
        evaluator.evaluate_incidental_scene_text();


    }

}


