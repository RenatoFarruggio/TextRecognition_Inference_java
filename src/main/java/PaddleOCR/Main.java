package PaddleOCR;

import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {

        //String imageName = "img_10.jpg"; // "Please lower your volume when you pass residental area"
        //String imageName = "img_11.jpg"; // "Beware of maintenance vehicles"
        String imageName = "img_1.jpg";

        Long start_load = System.currentTimeMillis();
        InferenceModel inference = new InferenceModel();
        Long end_load = System.currentTimeMillis();

        Long start_eval = System.currentTimeMillis();
        try {
            inference.inference_on_example_image(imageName);
        } catch (IOException | TranslateException e) {
            e.printStackTrace();
        }
        Long end_eval = System.currentTimeMillis();
        System.out.println("Load time: " + (end_load - start_load));
        System.out.println("Eval time: " + (end_eval - start_eval));


    }
}
