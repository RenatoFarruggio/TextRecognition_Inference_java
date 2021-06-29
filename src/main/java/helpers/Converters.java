package helpers;

import PaddleOCR.Textbox;
import ai.djl.modality.cv.output.DetectedObjects;

import java.util.ArrayList;
import java.util.List;

public class Converters {
    public static List<Textbox> DetectedObjects2ListOfTextboxes(DetectedObjects detectedObjects, int imageWidth, int imageHeight) {
        List<Textbox> textboxes = new ArrayList<>();
        for (int i = 0; i < detectedObjects.getNumberOfObjects(); i++) {
            DetectedObjects.DetectedObject det = detectedObjects.item(i);
            Textbox detectedTextbox = Textbox.fromDetectedObject_extended(det, imageWidth, imageHeight);
            textboxes.add(detectedTextbox);
        }
        return textboxes;
    }
}
