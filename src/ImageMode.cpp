#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include "ImageMode.h"
#include "Detect_and_Draw.h"


namespace fs = std::filesystem;

void image_mode(DetectAndDraw& detector, const std::string& imgDirectory) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(imgDirectory)) {
        std::string filePath = entry.path().string();
        if (filePath.find(".jpg") != std::string::npos || filePath.find(".png") != std::string::npos) {
            imagePaths.push_back(filePath);
        }
    }

    if (imagePaths.empty()) {
        std::cerr << "Image path is empty - Error!" << imgDirectory << "\n";
        return;
    }

    cv::namedWindow("YOLO Image Detection", cv::WINDOW_NORMAL);

    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Couldn't load image - Error!\n";
            continue;
        }

        std::vector<float> confidences;
        std::vector<int> classIds;
        std::vector<int> all_classes = {0, 1, 2};
        std::vector<cv::Rect> boxes = detector.detect_objects(image, classIds, confidences, all_classes);

        std::cout << "Objects found: " << boxes.size() << std::endl;
        if (boxes.empty()) continue;

        DetectAndDraw::draw_detections(image, boxes, classIds);

        cv::resize(image, image, cv::Size(), 0.5, 0.5);

        cv::imshow("YOLO Image Detection", image);

        std::cout << "Press any key to continue or ESC to finish the program.\n";
        int key = cv::waitKey(0);
        if (key == 27) break;
    }

    cv::destroyAllWindows();
}
