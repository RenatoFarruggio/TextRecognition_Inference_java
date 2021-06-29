package PaddleOCR;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A Textbox defines a textbox of a picture.
 * The values refer to the actual absolute pixel position on the picture.
 */
public class Textbox {
    int xMin, xMax;
    int yMin, yMax;

    private Textbox(int xMin, int xMax, int yMin, int yMax) {
        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
    }

    public static Textbox fromIncidentalSceneTextTXT(String lineOfTxtFile) {
        String[] stringValues = lineOfTxtFile.split(",");
        List<Integer> xValues = Arrays.asList(
                Integer.parseInt(stringValues[0]),
                Integer.parseInt(stringValues[2]),
                Integer.parseInt(stringValues[4]),
                Integer.parseInt(stringValues[6]));
        List<Integer> yValues = Arrays.asList(
                Integer.parseInt(stringValues[1]),
                Integer.parseInt(stringValues[3]),
                Integer.parseInt(stringValues[5]),
                Integer.parseInt(stringValues[7]));
        int xMin = Collections.min(xValues);
        int xMax = Collections.max(xValues);
        int yMin = Collections.min(yValues);
        int yMax = Collections.max(yValues);

        return new Textbox(xMin,xMax,yMin,yMax);
    }

    public static Textbox fromDetectedObject(DetectedObjects.DetectedObject detectedObject, int imageWidth, int imageHeight) {
        Rectangle rect = detectedObject.getBoundingBox().getBounds();

        int xMin = (int)(rect.getX() * imageWidth);
        int xMax = (int)(rect.getWidth() * imageWidth) + xMin;
        int yMin = (int)(rect.getY() * imageHeight);
        int yMax = (int)(rect.getHeight() * imageHeight) + yMin;
        assert xMin < xMax;
        assert yMin < yMax;

        return new Textbox(xMin,xMax,yMin,yMax);
    }

    public static Textbox fromDetectedObject_extended(DetectedObjects.DetectedObject detectedObject, int imageWidth, int imageHeight) {
        Rectangle rect = detectedObject.getBoundingBox().getBounds();

        double xmin = rect.getX();
        double ymin = rect.getY();
        double width = rect.getWidth();
        double height = rect.getHeight();

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


        int xMin = (int)(newX * imageWidth);
        int xMax = (int)(width * imageWidth) + xMin;
        int yMin = (int)(newY * imageHeight);
        int yMax = (int)(height * imageHeight) + yMin;
        assert xMin < xMax;
        assert yMin < yMax;

        return new Textbox(xMin,xMax,yMin,yMax);
    }



    public double area() {
        return (this.xMax - this.xMin) * (this.yMax - this.yMin);
    }

    public double getIoU(Textbox other) {
        if (this.xMax < other.xMin ||
            this.xMin > other.xMax ||
            this.yMax < other.yMin ||
            this.yMin > other.yMax) {
            return 0.0d;
        }

        int xMinIntersection = Math.max(this.xMin, other.xMin);
        int xMaxIntersection = Math.min(this.xMax, other.xMax);
        int yMinIntersection = Math.max(this.yMin, other.yMin);
        int yMaxIntersection = Math.min(this.yMax, other.yMax);

        double intersection = (xMaxIntersection-xMinIntersection) * (yMaxIntersection - yMinIntersection);

        double union = this.area() + other.area() - intersection;

        return intersection / union;
    }

    @Override
    public String toString() {
        return "Textbox {" +
                "xMin=" + xMin +
                ", xMax=" + xMax +
                ", yMin=" + yMin +
                ", yMax=" + yMax +
                '}';
    }

    /*
    public static void main(String[] args) {
        Textbox exampleTextbox = Textbox.fromIncidentalSceneTextTXT("822,288,872,286,871,298,823,300,yourself");
        System.out.println(exampleTextbox);
    }*/
}
