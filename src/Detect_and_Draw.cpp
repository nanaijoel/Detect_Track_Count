#include "Detect_and_Draw.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mutex>


extern std::mutex count_mutex;


cv::Mat preprocess_image(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0,
                           cv::Size(INP_WIDTH, INP_HEIGHT),
                           cv::Scalar(0, 0, 0), true, false);
    return blob;
}


cv::dnn::Net load_model(const std::string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Fehler beim Laden des Modells!\n";
        exit(-1);
    }
    return net;
}

std::vector<cv::Rect> detect_objects(cv::dnn::Net& net, const cv::Mat& image, std::vector<int>& classIds) {
    if (image.empty()) {
        std::cerr << "❌ Fehler: Eingabebild ist leer!\n";
        return {};
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INP_WIDTH, INP_HEIGHT), cv::Scalar(0, 0, 0), true, false);

    if (blob.empty()) {
        std::cerr << "❌ Fehler: Blob-Erstellung fehlgeschlagen!" << std::endl;
        return {};
    }

    net.setInput(blob);
    cv::Mat output = net.forward();

    int numDetections = output.size[2];
    int numFeatures = output.size[1];

    cv::Mat output2D = output.reshape(1, numFeatures).t();

    float x_factor = static_cast<float>(image.cols) / INP_WIDTH;
    float y_factor = static_cast<float>(image.rows) / INP_HEIGHT;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> detectedClassIds;

    for (int i = 0; i < numDetections; ++i) {
        const float* data = output2D.ptr<float>(i);
        cv::Mat scoresMat(1, 3, CV_32FC1, (void*)(data + 4));
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scoresMat, nullptr, &maxClassScore, nullptr, &classIdPoint);

        if (maxClassScore > CONF_THRESHOLD) {
            int classId = classIdPoint.x;
            float centerX = data[0] * x_factor;
            float centerY = data[1] * y_factor;
            float width = data[2] * x_factor;
            float height = data[3] * y_factor;
            int left = std::max(0, static_cast<int>(centerX - width / 2));
            int top = std::max(0, static_cast<int>(centerY - height / 2));

            boxes.emplace_back(left, top, static_cast<int>(width), static_cast<int>(height));
            scores.push_back(static_cast<float>(maxClassScore));
            detectedClassIds.push_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<cv::Rect> filteredBoxes;
    std::vector<int> filteredClassIds;

    for (int idx : indices) {
        filteredBoxes.push_back(boxes[idx]);
        filteredClassIds.push_back(detectedClassIds[idx]);
    }

    classIds = filteredClassIds;
    return filteredBoxes;
}

void draw_detections(cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<int>& classIds) {
    std::cout << "🔍 Zeichne " << boxes.size() << " Objekte...\n";

    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 200), 3);

        std::string classname = (classIds[i] == 0) ? "Bear" : (classIds[i] == 1) ? "Frog" : "Cola";

        int baseLine;
        cv::Size labelSize = cv::getTextSize(classname, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseLine);
        int textX = boxes[i].x + (boxes[i].width - labelSize.width) / 2;
        int textY = std::max(boxes[i].y - 10, labelSize.height);

        cv::putText(image, classname, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }
}


cv::Mat create_info_panel(int height) {
    cv::Mat info_panel(height, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(info_panel, "OBJECT COUNTS", cv::Point(20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 55, 255), 3);

    std::lock_guard<std::mutex> lock(count_mutex);
    for (int i = 0; i < 3; i++) {
        std::string classname = (i == 0) ? "Bear" : (i == 1) ? "Frog" : "Cola";
        cv::putText(info_panel, classname + " Actual: " + std::to_string(actual_counts[i]) +
                                  " | Total: " + std::to_string(total_counts[i]),
                    cv::Point(20, 100 + i * 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
    return info_panel;
}