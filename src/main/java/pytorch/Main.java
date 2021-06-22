// https://www.programmersought.com/article/86276896101/
package pytorch;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.repository.zoo.*;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import javax.imageio.ImageIO;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.nio.file.FileSystems;
import java.nio.file.Path;


class Main {
    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        //run_orig();
        //run_custom();
        run_tensorflow();
        //run_pretrained();
    }

    /**
     * Word recognition
     */
    private static void run_paddlepaddle() throws IOException, ModelException, TranslateException {
        System.out.println("Test1");

        //String modelFolder = "models/";
        //String modelName = "frozen_model.pb";
        //URI modelUri = new File(modelFolder).toURI();
        //Path modelPath = Path.of(modelUri);
        //URL modelUrl = modelUri.toURL();

        //System.out.println("URI: " + modelUri);

        //System.out.println("Absolute path of model: " + modelPath.toAbsolutePath());


        var criteria3 = Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, String.class)
                .optModelUrls("https://resources.djl.ai/test-models/paddleOCR/mobile/rec_crnn.zip")
                //.optTranslator(new PpWordRecognitionTranslator())
                .build();
        var recognitionModel = ModelZoo.loadModel(criteria3);
        var recognizer = recognitionModel.newPredictor();

//        Translator<Image, DetectedObjects> translator = SingleShotDetectionTranslator.builder()
//                .addTransform(new ToTensor())
//                .optSynsetUrl("https://mysynset.txt")
//                .build();

//        Criteria<Image, DetectedObjects> criteria =
//                Criteria.builder()
//                        .setTypes(Image.class, DetectedObjects.class)
//                        .optApplication(Application.CV.WORD_RECOGNITION)
//                        //.optModelUrls("file:///Users/home/mxnet/vgg16_atrous_custom")
//                        .optTranslator(translator)
//                        .build();


//        Criteria<Image, DetectedObjects> criteria =
//                Criteria.builder()
//                        .optApplication(Application.CV.WORD_RECOGNITION)
//                        .setTypes(Image.class, DetectedObjects.class)
//                        //.optModelUrls(modelUri.toString())
//                        //.optFilter("backbone", "mobilenet_v2")
//                        .optDevice(Device.cpu())
//                        .optTranslator(translator)
//                        .optEngine("TensorFlow")
//                        //.optOption("Tags", "")
//                        //.optOption("SignatureDefKey", "default")
//                        .optProgress(new ProgressBar())
//                        .build();


        System.out.println("Listing models...");
        System.out.println("Number of models found: " + ModelZoo.listModels(criteria3).size());
        System.out.println(ModelZoo.listModels(criteria3));
        System.out.println("Done.");

        ZooModel<?, ?> model = ModelZoo.loadModel(criteria3);
        System.out.println("Input: " + model.describeInput());
        System.out.println("Output: " + model.describeOutput());
        //ZooModel<Image, DetectedObjects> model = criteria.getModelZoo();


        //System.out.println("Model: " + model);
        System.out.println("Test2");

    }

    private static void run_tensorflow() throws IOException, ModelException, TranslateException {
        System.out.println("Test1");

        String modelFolder = "models/";
        String modelName = "frozen_model.pb";
        URI modelUri = new File(modelFolder).toURI();
        Path modelPath = Path.of(modelUri);
        URL modelUrl = modelUri.toURL();

        System.out.println("URI: " + modelUri);

        System.out.println("Absolute path of model: " + modelPath.toAbsolutePath());

        //Translator<ai.djl.modality.cv.Image, DetectedObjects> translator;
        SingleShotDetectionTranslator translator;

        translator = SingleShotDetectionTranslator.builder().setPipeline(new Pipeline()).build();

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.WORD_RECOGNITION)
                        .setTypes(Image.class, DetectedObjects.class)
                        //.optModelUrls(modelUri.toString())
                        //.optFilter("backbone", "mobilenet_v2")
                        .optDevice(Device.cpu())
                        .optTranslator(translator)
                        .optEngine("TensorFlow")
                        //.optOption("Tags", "")
                        //.optOption("SignatureDefKey", "default")
                        .optProgress(new ProgressBar())
                        .build();


        System.out.println("Listing models...");
        System.out.println("Number of models found: " + ModelZoo.listModels(criteria).size());
        System.out.println(ModelZoo.listModels(criteria));
        System.out.println("Done.");

        ZooModel<?, ?> model = ModelZoo.loadModel(criteria);
        System.out.println("Input: " + model.describeInput());
        System.out.println("Output: " + model.describeOutput());
        //ZooModel<Image, DetectedObjects> model = criteria.getModelZoo();


        //System.out.println("Model: " + model);
        System.out.println("Test2");

    }

    private static void run_custom() throws IOException, ModelException, TranslateException {
        System.out.println("Test1");

        String modelFolder = "models/";
        String modelName = "final.pth";
        URI modelUri = new File(modelFolder + modelName).toURI();
        Path modelPath = Path.of(modelUri);
        URL modelUrl = modelUri.toURL();

        System.out.println("URI: " + modelUri.toString());

        System.out.println("Absolute path of model: " + modelPath.toAbsolutePath());

        //Translator<ai.djl.modality.cv.Image, DetectedObjects> translator;
        SingleShotDetectionTranslator translator;
        translator = SingleShotDetectionTranslator.builder().setPipeline(new Pipeline()).build();

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        //.optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        //.optModelUrls(modelUri.toString())
                        //.optFilter("backbone", "mobilenet_v2")
                        .optDevice(Device.cpu())
                        .optTranslator(translator)
                        .optEngine("PyTorch")
                        //.optOption("Tags", "")
                        //.optOption("SignatureDefKey", "default")
                        //.optProgress(new ProgressBar())
                        .build();


        ModelZoo.loadModel(criteria);
        //ZooModel<Image, DetectedObjects> model = criteria.getModelZoo();

        System.out.println("Listing models...");
        System.out.println(ModelZoo.listModels());
        System.out.println("Done.");

        //System.out.println("Model: " + model);
        System.out.println("Test2");
    }

    /*
    private static void run_custom() {
        // Read a picture
        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        //BufferedImage img2 = BufferedImageUtils.fromUrl(url);
        BufferedImage img = ImageHandler.load_image("example_1.png");

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        // Create a model object from local pth file
        String modelPath = "models";
        //    String modelName = "final.pt";

        //URI modelUri = new File(modelPath + "/" + modelName).toURI();
        URI modelUri = new File(modelPath).toURI();


        //Path modelDir = Paths.get(modelUri);
        Path modelDir = FileSystems.getDefault().getPath(modelPath).toAbsolutePath();

        System.out.println("Model directory: " + modelDir);
        System.out.println("ModelURI: " + modelDir.toUri());

        Model ptModel = Model.newInstance(Device.cpu());
        ptModel.load(modelDir);

        //  URI modelUri = classLoader.getResource(modelPath).toURI();
        //  Path modelDir = Paths.get(modelUri);
        //  System.out.println("Model directory: " + modelDir);
        //  //Model ptModel = Model.newInstance("my_model", Device.cpu());
        //  Model ptModel = Model.newInstance(Device.cpu());
        //  ptModel.load(modelDir, "my_model");
    }*/

    private static void run_pretrained() throws ModelNotFoundException, IOException, MalformedModelException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, Classifications.class)
                        .optFilter("layer", "50")
                        .optFilter("flavor", "v1")
                        .optFilter("dataset", "cifar10")
                        .build();

        ZooModel<Image, Classifications> ssd = ModelZoo.loadModel(criteria);
    }

    private static void run_orig() throws IOException, ModelException, TranslateException {
        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        //BufferedImage img = BufferedImageUtils.fromUrl(url);
        BufferedImage img = ImageIO.read(new URL(url));

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                System.out.println(detection);
            }
        }


        //   listModels() mit modelzoo 0.4.0
        //  {
        //  ai.djl.Application@142269f2=[ai.djl.pytorch:resnet:0.0.1 {"layers":"50","dataset":"imagenet"}, ai.djl.pytorch:resnet:0.0.1 {"layers":"18","dataset":"imagenet"}],
        //  ai.djl.Application@331acdad=[ai.djl.pytorch:ssd:0.0.1 {"size":"300","backbone":"resnet50","dataset":"coco"}]
        //  }

        //   listModels() mit modelzoo 0.11.0
        //  {CV.IMAGE_CLASSIFICATION=[
        //  ai.djl.pytorch:resnet:0.0.1 {"layers":"50","dataset":"imagenet"}, ai.djl.pytorch:resnet:0.0.1 {"layers":"18","dataset":"imagenet"}],
        //  CV.OBJECT_DETECTION=[ai.djl.pytorch:ssd:0.0.1 {"size":"300","backbone":"resnet50","dataset":"coco"}],
        //  NLP.QUESTION_ANSWER=[ai.djl.pytorch:bertqa:0.0.1 {"backbone":"bert","dataset":"SQuAD"}, ai.djl.pytorch:bertqa:0.0.1 {"backbone":"bert","dataset":"SQuAD","cased":"true"}, ai.djl.pytorch:bertqa:0.0.1 {"backbone":"distilbert","dataset":"SQuAD","cased":"true"}],
        //  NLP.SENTIMENT_ANALYSIS=[ai.djl.pytorch:distilbert:0.0.1 {"backbone":"distilbert","dataset":"sst"}]}
    }
}
