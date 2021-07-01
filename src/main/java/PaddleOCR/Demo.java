package PaddleOCR;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Objects;

import ai.djl.translate.TranslateException;
import helpers.ImageHandler;

public class Demo {

    private static File selectedFolder;
    private static InferenceModel inferenceModel;

    /**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event-dispatching thread.
     */
    private static void createAndShowGUI() {
        //Make sure we have nice window decorations.
        //JFrame.setDefaultLookAndFeelDecorated(true);

        //Create and set up the window.
        JFrame frame = new JFrame("Demo: Scene Text Recognition");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //BorderLayout borderLayout = new BorderLayout();
        BoxLayout boxLayout = new BoxLayout(frame.getContentPane(), BoxLayout.X_AXIS);
        frame.getContentPane().setLayout(boxLayout);

        JLabel picture1label = new JLabel(new ImageIcon(ImageHandler.loadImage("example_images/img_1.jpg").getScaledInstance(300, 200, Image.SCALE_SMOOTH)));
        JLabel picture2label = new JLabel(new ImageIcon(ImageHandler.loadImage("example_images/img_1.jpg").getScaledInstance(300, 200, Image.SCALE_SMOOTH)));

        final JLabel statsLabelStatus = new JLabel("Finished.");
        final JLabel statsLabelImageName = new JLabel("Image name: ");

        // Load inference model
        String pathDetectionModel = "models/det_db.zip";
        String pathRecognitionModel = "models/rec_crnn.zip";
        File detectionModel = new File(pathDetectionModel);
        File recognitionModel = new File(pathRecognitionModel);
        inferenceModel = new InferenceModel(detectionModel, recognitionModel);

        // Images list
        DefaultListModel<String> filesList = new DefaultListModel<>();
        JList<String> jlist = new JList<>(filesList);
        jlist.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                super.mouseClicked(e);
                String item = jlist.getSelectedValue();
                System.out.println(item);
                if (item != null) {
                    File file = new File(selectedFolder, item);
                    // Load Image
                    BufferedImage img = ImageHandler.loadImage(file.getPath());
                    double scaleFactor = 1.0D / Math.max(img.getWidth() / (double)600, img.getHeight() / (double)600);
                    int scaledWidth = (int)(img.getWidth() * scaleFactor);
                    int scaledHeight = (int)(img.getHeight() * scaleFactor);
                    picture1label.setIcon(new ImageIcon(img.getScaledInstance(scaledWidth, scaledHeight, Image.SCALE_SMOOTH)));
                    Thread thread = new Thread(){
                        @Override
                        public void run() {

                            SwingUtilities.invokeLater(() -> statsLabelStatus.setText("Loading..."));

                            BufferedImage result = null;
                            try {
                                result = inferenceModel.inference_on_example_image(file.getPath());
                            } catch (IOException | TranslateException e) {
                                e.printStackTrace();
                            }
                            if (result != null) {
                                BufferedImage resultingImage = result;
                                SwingUtilities.invokeLater(() -> {
                                    picture2label.setIcon(new ImageIcon(resultingImage.getScaledInstance(scaledWidth, scaledHeight, Image.SCALE_SMOOTH)));
                                    statsLabelImageName.setText("Image name: " + file.getName());
                                    statsLabelStatus.setText("Finished.");
                                });
                            }
                            System.out.println("Done.");
                        }
                    };
                    thread.start();
                }
            }
        });
        JScrollPane scrollPane = new JScrollPane(jlist);

        JButton buttonSelectFolder = new JButton();
        buttonSelectFolder.setAction(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();
                fileChooser.setCurrentDirectory(new File("."));
                fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                fileChooser.showOpenDialog(null);
                selectedFolder = fileChooser.getSelectedFile();
                filesList.clear();
                for (File file : Objects.requireNonNull(selectedFolder.listFiles(f -> (f.getPath().endsWith(".jpg") || f.getPath().endsWith(".png"))))) {
                    filesList.addElement(file.getName());
                }

            }
        });
        buttonSelectFolder.setText("Select Folder");


        JPanel scrollpaneframe = new JPanel();
        scrollpaneframe.setLayout(new BoxLayout(scrollpaneframe, BoxLayout.Y_AXIS));
        scrollpaneframe.add(scrollPane);
        scrollpaneframe.add(buttonSelectFolder);

        frame.getContentPane().add(scrollpaneframe, BorderLayout.WEST);

        // Info Panel
        JPanel infoPanel = new JPanel();
        infoPanel.setLayout(new BoxLayout(infoPanel, BoxLayout.Y_AXIS));

        JPanel picturesPanel = new JPanel();
        picturesPanel.setLayout(new BoxLayout(picturesPanel, BoxLayout.X_AXIS));

        // Picture 1
        JPanel picture1frame = new JPanel();
        picture1frame.setLayout(new BoxLayout(picture1frame, BoxLayout.Y_AXIS));
        picture1frame.add(new JLabel("Original"));
        picture1frame.add(picture1label);

        picturesPanel.add(picture1frame, BorderLayout.WEST);

        // Picture 2
        JPanel picture2frame = new JPanel();
        picture2frame.setLayout(new BoxLayout(picture2frame, BoxLayout.Y_AXIS));
        picture2frame.add(new JLabel("Recognized Text"));
        picture2frame.add(picture2label);

        picturesPanel.add(picture2frame, BorderLayout.EAST);


        infoPanel.add(picturesPanel);

        JPanel statsPanel = new JPanel();
        statsPanel.setLayout(new GridLayout(1,2));
        statsPanel.add(statsLabelStatus, BorderLayout.WEST);
        statsPanel.add(statsLabelImageName, BorderLayout.WEST);

        infoPanel.add(statsPanel);
        frame.getContentPane().add(infoPanel, BorderLayout.EAST);

        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });

        try {
            inferenceModel.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}