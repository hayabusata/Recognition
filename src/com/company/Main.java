package com.company;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    public static Mat colorToGray(Mat image) {
        Mat grayImage = new Mat(image.rows(), image.cols(), image.type());
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGRA2GRAY);
        Core.normalize(grayImage, grayImage, 0, 255, Core.NORM_MINMAX);
        return grayImage;
    }


    public static void main(String[] args) {
        // write your code here
        long start = System.currentTimeMillis();

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String objectFileName = "IMG_0977.JPG";
        String sceneFileName = "IMG_0978.JPG";

        Mat objectImage = Imgcodecs.imread(objectFileName);
        Mat sceneImage = Imgcodecs.imread(sceneFileName);
        if (objectImage == null || sceneImage == null) {
            System.out.println("image file is not found");
            System.exit(0);
        }

        Mat grayObjectImage = colorToGray(objectImage);
        Mat graySceneImage = colorToGray(sceneImage);

        Imgcodecs.imwrite("grayObject.jpg", grayObjectImage);
        Imgcodecs.imwrite("grayScene.jpg", graySceneImage);


        // ---------------AKAZE--------------

        FeatureDetector akazeDetector = FeatureDetector.create(FeatureDetector.AKAZE);
        DescriptorExtractor akazeExtractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);

        MatOfKeyPoint objectKeyPoint = new MatOfKeyPoint();
        akazeDetector.detect(grayObjectImage, objectKeyPoint);

        MatOfKeyPoint sceneKeyPoint = new MatOfKeyPoint();
        akazeDetector.detect(graySceneImage, sceneKeyPoint);

        Mat objectDescripters = new Mat(objectImage.rows(), objectImage.cols(), objectImage.type());
        akazeExtractor.compute(grayObjectImage, objectKeyPoint, objectDescripters);

        Mat sceneDescripters = new Mat(sceneImage.rows(), sceneImage.cols(), sceneImage.type());
        akazeExtractor.compute(graySceneImage, sceneKeyPoint, sceneDescripters);

        MatOfDMatch matchs = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        matcher.match(objectDescripters, sceneDescripters, matchs);

        int N = 50;
        DMatch[] tmp1 = matchs.toArray();
        DMatch[] tmp2 = new DMatch[N];
        for (int i = 0; i < tmp2.length; i++) {
            tmp2[i] = tmp1[i];
        }
        matchs.fromArray(tmp2);

        Mat matchedImage = new Mat(objectImage.rows(), objectImage.cols() * 2, objectImage.type());
        Features2d.drawMatches(objectImage, objectKeyPoint, sceneImage, sceneKeyPoint, matchs, matchedImage);

        Imgcodecs.imwrite("descriptedImageByAKAZE.jpg", matchedImage);
        //Imgcodecs.imwrite("sameOutputImageByAKAZE.jpg", matchedImage);

        long end = System.currentTimeMillis();
        System.out.println(end - start);
    }

}
