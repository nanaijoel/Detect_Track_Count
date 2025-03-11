#include "ImageMode.h"
#include "Detect_and_Draw.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void image_mode(cv::dnn::Net& net, const std::string& imgDirectory) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(imgDirectory)) {
        std::string filePath = entry.path().string();
        if (filePath.find(".jpg") != std::string::npos || filePath.find(".png") != std::string::npos) {
            imagePaths.push_back(filePath);
        }
    }

    if (imagePaths.empty()) {
        std::cerr << "❌ Keine Bilder gefunden im Verzeichnis: " << imgDirectory << "\n";
        return;
    }

    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "⚠ Fehler: Bild konnte nicht geladen werden!\n";
            continue;
        }

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detect_objects(net, image, classIds);

        std::cout << "🎯 Gefundene Objekte: " << boxes.size() << std::endl;
        if (boxes.empty()) continue;

        draw_detections(image, boxes, classIds);

        cv::resize(image, image, cv::Size(800, 600));
        cv::imshow("YOLO Image Detection", image);

        std::cout << "🔹 Drücke eine Taste für das nächste Bild oder ESC zum Beenden\n";
        int key = cv::waitKey(0);
        if (key == 27) break;
    }

    cv::destroyAllWindows();
}
