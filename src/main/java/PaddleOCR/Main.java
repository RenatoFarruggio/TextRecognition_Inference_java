package PaddleOCR;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {

        String imageName = "img_5.png";

        Inference_v1 inferenceV1 = new Inference_v1();
        try {
            inferenceV1.do_stuff(imageName);
        } catch (MalformedModelException e) {
            e.printStackTrace();
        } catch (ModelNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (TranslateException e) {
            e.printStackTrace();
        }

    }



}
