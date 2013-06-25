import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.GenericDescriptorMatcher;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


public class HelloOpenCV {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		new DetectFaceDemo().run();

		// 比較画像01
		Mat image01 = Highgui.imread("./resources/test01.jpg"); // ←ここのファイルを書き換えればおｋ

		// 比較画像02
		Mat image02 = Highgui.imread("./resources/test02.jpg"); // ←ここのファイルを書き換えればおｋ
		if (image01 == null || image02 == null) {
			System.out.println("ﾇﾙﾎﾟ('A`)");
			System.exit(0);
		}
		
		Mat grayImage01 = new Mat(image01.rows(), image01.cols(), image01.type());
		Imgproc.cvtColor(image01, grayImage01, Imgproc.COLOR_BGRA2GRAY);
		Core.normalize(grayImage01, grayImage01, 0, 255, Core.NORM_MINMAX);

		Mat	grayImage02 = new Mat(image02.rows(), image02.cols(), image02.type());
		Imgproc.cvtColor(image02, grayImage02, Imgproc.COLOR_BGRA2GRAY);
		Core.normalize(grayImage02, grayImage02, 0, 255, Core.NORM_MINMAX);
		
		FeatureDetector siftDetector = FeatureDetector.create(FeatureDetector.SIFT);
		DescriptorExtractor siftExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
		
		MatOfKeyPoint keyPoint01 = new MatOfKeyPoint();
		siftDetector.detect(grayImage01, keyPoint01);

		MatOfKeyPoint keyPoint02 = new MatOfKeyPoint();
		siftDetector.detect(grayImage02, keyPoint02);

		Mat descripters01 = new Mat(image01.rows(), image01.cols(), image01.type());
		siftExtractor.compute(grayImage01, keyPoint01, descripters01);
		
		Mat descripters02 = new Mat(image02.rows(), image02.cols(), image02.type());
		siftExtractor.compute(grayImage02, keyPoint02, descripters02);
		
		MatOfDMatch matchs = new MatOfDMatch();
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
		matcher.match(descripters01, descripters02, matchs);
		
		int N = 50;
		DMatch[] tmp01 = matchs.toArray();
		DMatch[] tmp02 = new DMatch[N];
		for (int i=0; i<tmp02.length; i++) {
			tmp02[i] = tmp01[i];
		}
		matchs.fromArray(tmp02);
		
		int year = Calendar.getInstance().get(Calendar.YEAR);
		int month = Calendar.getInstance().get(Calendar.MONTH) + 1;
		int day = Calendar.getInstance().get(Calendar.DAY_OF_MONTH);
		int hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY);
		int minute = Calendar.getInstance().get(Calendar.MINUTE);
		int second = Calendar.getInstance().get(Calendar.SECOND);
		String now = year + "" + month + "" + day + "" + hour + "" + minute + "" + second;
		
		Mat matchedImage = new Mat(image01.rows(), image01.cols()*2, image01.type());
		Features2d.drawMatches(image01, keyPoint01, image02, keyPoint02, matchs, matchedImage);

		// 出力画像 at SIFT
		Highgui.imwrite("./resources/descriptedImageBySIFT-" + now + ".jpg", matchedImage);
		
		FeatureDetector surfDetector = FeatureDetector.create(FeatureDetector.SURF);
		DescriptorExtractor surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

		keyPoint01 = new MatOfKeyPoint();
		surfDetector.detect(grayImage01, keyPoint01);

		keyPoint02 = new MatOfKeyPoint();
		surfDetector.detect(grayImage02, keyPoint02);

		descripters01 = new Mat(image01.rows(), image01.cols(), image01.type());
		surfExtractor.compute(grayImage01, keyPoint01, descripters01);
		
		descripters02 = new Mat(image02.rows(), image02.cols(), image02.type());
		surfExtractor.compute(grayImage02, keyPoint02, descripters02);
		
		matchs = new MatOfDMatch();
		matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
		matcher.match(descripters01, descripters02, matchs);
		
		DMatch[] tmp03 = matchs.toArray();
		DMatch[] tmp04 = new DMatch[N];
		for (int i=0; i<tmp04.length; i++) {
			tmp04[i] = tmp03[i];
		}
		matchs.fromArray(tmp02);
				
		matchedImage = new Mat(image01.rows(), image01.cols()*2, image01.type());
		Features2d.drawMatches(image01, keyPoint01, image02, keyPoint02, matchs, matchedImage);

		// 出力画像 at SURF
		Highgui.imwrite("./resources/descriptedImageBySURF-" + now + ".jpg", matchedImage);
	}
}

//class DetectFaceDemo {
//	  public void run() {
//	    System.out.println("\nRunning DetectFaceDemo");
//
//	    // Create a face detector from the cascade file in the resources
//	    // directory.
//	    CascadeClassifier faceDetector = new CascadeClassifier("data/lbpcascades/lbpcascade_frontalface.xml");
//	    Mat image = Highgui.imread("./resources/lena2.png");
//
//	    // Detect faces in the image.
//	    // MatOfRect is a special container class for Rect.
//	    MatOfRect faceDetections = new MatOfRect();
//	    faceDetector.detectMultiScale(image, faceDetections);
//
//	    System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
//
//	    // Draw a bounding box around each face.
//	    for (Rect rect : faceDetections.toArray()) {
//	        Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
//	    }
//
//	    // Save the visualized detection.
//	    String filename = "faceDetection.png";
//	    System.out.println(String.format("Writing %s", filename));
//	    Highgui.imwrite(filename, image);
//  }
//
//  public byte[] getImageBytesForBufferedImageOfDataTypeSize1(BufferedImage image) {
//			int[] pixels = new int[image.getWidth()*image.getHeight()];
//			byte[] bytes = new byte[image.getWidth()*image.getHeight()*3];
//			DataBufferInt buffer = (DataBufferInt)(image.getRaster().getDataBuffer());
//			
//			int j =0;
//			for(int i = 0; i < image.getWidth()*image.getHeight(); i++){
//				pixels[i] = buffer.getElem(i);
//				
//				bytes[j] =   (byte)(0xFF & ((pixels[i] & 0x000000FF) >> 0)); //r
//				bytes[j+1] = (byte)(0xFF & ((pixels[i] & 0x0000FF00) >> 8)); //g
//				bytes[j+2] = (byte)(0xFF & ((pixels[i] & 0x00FF0000) >> 16)); //b
//				j+=3;
//			}
//			return bytes;
//	}
//  
// // +++USAGE+++ 
////  pixels = getImageBytesForBufferedImageOfDataTypeSize1(bufferedImage);
////  Mat mat = new Mat(img.getHeight(),img.getWidth(),CvType.CV_8UC3);
////  mat.put(0, 0, pixels);
//}