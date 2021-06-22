package PaddleOCR;

import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.paddlepaddle.zoo.cv.imageclassification.PpWordRotateTranslator;
import ai.djl.paddlepaddle.zoo.cv.objectdetection.PpWordDetectionTranslator;
import ai.djl.paddlepaddle.zoo.cv.wordrecognition.PpWordRecognitionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.TranslateException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/* *********************** */
/*  Heavily influenced by  */
/* http://docs.djl.ai/jupyter/paddlepaddle/paddle_ocr_java.html#word-recgonition-model
 */

public class Inference_v1 {

    public void do_stuff(String imageName) throws MalformedModelException, ModelNotFoundException, IOException, TranslateException {
        // Load image
        //String url = "https://resources.djl.ai/images/flight_ticket.jpg";
        //Image img = ImageFactory.getInstance().fromUrl(url);
        String imagesFolder = "example_images/";
        //String imageName = "img_11.jpg";
        //String imageName = "img_5.png";
        Image img = ImageFactory.getInstance().fromFile(Path.of(new File(imagesFolder + imageName).toURI()));
        img.getWrappedImage();

        img.save(new FileOutputStream("output_0_original.png"), "png");

        // Load text detection model
        var criteria1 = Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/det_db.zip")
                .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<String, String>()))
                .build();
        var detectionModel = ModelZoo.loadModel(criteria1);
        var detector = detectionModel.newPredictor();

        // Text detection
        var detectedObj = detector.predict(img);
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detectedObj);
        newImage.getWrappedImage();

        newImage.save(new FileOutputStream("output_1_textboxes.png"), "png");

        // Cut out single textboxes
        List<DetectedObjects.DetectedObject> boxes = detectedObj.items();

        // Load model for text orientation
        var criteria2 = Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/cls.zip")
                .optTranslator(new PpWordRotateTranslator())
                .build();
        var rotateModel = ModelZoo.loadModel(criteria2);
        var rotateClassifier = rotateModel.newPredictor();

        // Load model for text recognition
        var criteria3 = Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, String.class)
                .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/rec_crnn.zip")
                .optTranslator(new PpWordRecognitionTranslator())
                .build();
        var recognitionModel = ModelZoo.loadModel(criteria3);
        var recognizer = recognitionModel.newPredictor();

        // Evaluate the whole picture
        List<String> names = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> rect = new ArrayList<>();

        for (int i = 0; i < boxes.size(); i++) {
            Image subImg = getSubImage(img, boxes.get(i).getBoundingBox());
            if (subImg.getHeight() * 1.0 / subImg.getWidth() > 1.5) {
                subImg = rotateImg(subImg);
            }
            Classifications.Classification result = rotateClassifier.predict(subImg).best();
            if ("Rotate".equals(result.getClassName()) && result.getProbability() > 0.8) {
                subImg = rotateImg(subImg);
            }
            String name = recognizer.predict(subImg);
            names.add(name);
            prob.add(-1.0);
            rect.add(boxes.get(i).getBoundingBox());

            printWordAndBoundingBox(name, boxes.get(i).getBoundingBox());
        }
        newImage.drawBoundingBoxes(new DetectedObjects(names, prob, rect));
        newImage.getWrappedImage();

        newImage.save(new FileOutputStream("output_2_evaluated.png"), "png");
    }

    Image rotateImg(Image image) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
            return ImageFactory.getInstance().fromNDArray(rotated);
        }
    }

    Image getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
        int width = img.getWidth();
        int height = img.getHeight();
        int[] recovered = {
                (int) (extended[0] * width),
                (int) (extended[1] * height),
                (int) (extended[2] * width),
                (int) (extended[3] * height)
        };
        return img.getSubimage(recovered[0], recovered[1], recovered[2], recovered[3]);
    }

    double[] extendRect(double xmin, double ymin, double width, double height) {
        double centerx = xmin + width / 2;
        double centery = ymin + height / 2;
        if (width > height) {
            width += height * 2.0;
            height *= 3.0;
        } else {
            height += width * 2.0;
            width *= 3.0;
        }
        double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
        double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
        double newWidth = newX + width > 1 ? 1 - newX : width;
        double newHeight = newY + height > 1 ? 1 - newY : height;
        return new double[] {newX, newY, newWidth, newHeight};
    }

    void printWordAndBoundingBox(String word, BoundingBox box) {
        System.out.println("{word:" + word + ",\tbox:" + box + "}");
    }

}
