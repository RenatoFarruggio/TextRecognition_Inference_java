package helpers;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import javax.imageio.ImageIO;

public class ImageHandler {
    public static BufferedImage loadImage(String filename) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filename));
        } catch (IOException e) {
            System.out.println("Image " + filename + " not found!");
        }
        return img;
    }

    public static void saveImage(BufferedImage img) {
        saveImage(img,"", "saved", "png");
    }

    public static void saveImage(BufferedImage img, String name, String filetype) {
        saveImage(img, "", name, filetype);
    }

    public static void saveImage(BufferedImage img, String pathname, String name, String filetype) {
        if (!pathname.endsWith("/") & !pathname.equals(""))
            pathname += "/";
        // TODO: create output directory if not existent.
        //  Concrete: It will fail, if there is not "evaluations" folder!

        try {
            // retrieve image
            File outputfile = new File(pathname + name + "." + filetype);
            ImageIO.write(img, filetype, outputfile);
        } catch (IOException e) {
            System.out.println("Error while saving image: " + name + "\\." + filetype);
        }
    }

    public static void drawRectangle(BufferedImage img, int xmin, int xmax, int ymin, int ymax, Color color) {
        Graphics2D graph = img.createGraphics();
        graph.setColor(color);
        graph.setStroke(new BasicStroke(2));
        graph.drawRect(xmin, ymin, xmax-xmin, ymax-ymin);
        graph.dispose();
    }

    public static void drawRectangleRelative(BufferedImage img, double xmin, double xmax, double ymin, double ymax, Color color) {
        int img_width = img.getWidth();
        int img_height = img.getHeight();
        assert xmax + ymax <= 2;  // efficient way of checking
        // that the values are probably in the interval [0,1].
        xmin *= img_width;
        xmax *= img_width;
        ymin *= img_height;
        ymax *= img_height;
        drawRectangle(img, (int) xmin, (int) xmax, (int) ymin, (int) ymax, color);
    }

    public static void drawText(BufferedImage img, String str, int startposX, int startposY, Color color) {
        // TODO: add background color
        Graphics2D graph = img.createGraphics();
        graph.setColor(color);
        graph.drawString(str, startposX, startposY);
        graph.dispose();
    }
}