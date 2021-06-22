package pytorch;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.border.StrokeBorder;

public class ImageHandler {
    public static BufferedImage load_image(String filename) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filename));
        } catch (IOException e) {
            System.out.println("Image " + filename + " not found!");
        }
        return img;
    }

    public static void save_image(BufferedImage img) {
        try {
            // retrieve image
            File outputfile = new File("saved.png");
            ImageIO.write(img, "png", outputfile);
        } catch (IOException e) {
            System.out.println("Error while saving image!");
        }
    }

    public static BufferedImage draw_rectangle(BufferedImage img, int xmin, int xmax, int ymin, int ymax, Color color) {
        Graphics2D graph = img.createGraphics();
        graph.setColor(color);
        graph.setStroke(new BasicStroke(5));
        graph.drawRect(xmin, ymin, xmax-xmin, ymax-ymin);
        graph.dispose();
        return img;
    }

    public static BufferedImage draw_rectangle_relative(BufferedImage img, double xmin, double xmax, double ymin, double ymax, Color color) {
        int img_width = img.getWidth();
        int img_height = img.getHeight();
        xmin *= img_width;
        xmax *= img_width;
        ymin *= img_height;
        ymax *= img_height;
        return draw_rectangle(img, (int) xmin, (int) xmax, (int) ymin, (int) ymax, color);
    }
}