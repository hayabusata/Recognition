package com.company;

import org.opencv.core.Mat;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;

import static org.opencv.imgcodecs.Imgcodecs.imread;

/**
 * Created by syunta on 2017/06/15.
 */
public class Compression {
    public static void compressImage(String filename) throws IOException {
        File input = new File(filename);
        BufferedImage image = ImageIO.read(input);

        File compressedImageFile = new File("compressScene.jpg");
        OutputStream os = new FileOutputStream(compressedImageFile);

        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");
        ImageWriter writer = (ImageWriter)writers.next();

        ImageOutputStream ios = ImageIO.createImageOutputStream(os);
        writer.setOutput(ios);

        ImageWriteParam param = writer.getDefaultWriteParam();

        param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        param.setCompressionQuality(0.1f);
        writer.write(null, new IIOImage(image, null, null), param);

        os.close();
        ios.close();
        writer.dispose();
    }

    public static void main(String[] args) throws IOException {
        Compression.compressImage("sceneDedenne.JPG");
    }
}
