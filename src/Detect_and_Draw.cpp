#include <iostream>
#include "Detect_And_Draw.h"
#include "TotalCounter.h"
#include "CameraMode.h"

std::mutex count_mutex;

cv::Mat shared_frame;

DetectAndDraw::DetectAndDraw(const std::string& model_path) {
    net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Model loading failed - Error!\n";
        exit(-1);
    }
}

cv::Mat DetectAndDraw::preprocess_image(const cv::Mat& image) const {
    if (image.empty()) {
        std::cerr << "Image is empty - Error!\n";
        return {};
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0,
                           cv::Size(INP_WIDTH, INP_HEIGHT),
                           cv::Scalar(0, 0, 0), true, false);
    return blob;
}

cv::Rect DetectAndDraw::compute_bounding_box(const float* data, float x_factor, float y_factor) {
    float centerX = data[0] * x_factor;
    float centerY = data[1] * y_factor;
    float width = data[2] * x_factor;
    float height = data[3] * y_factor;
    int left = std::max(0, static_cast<int>(centerX - width / 2));
    int top = std::max(0, static_cast<int>(centerY - height / 2));
    return {left, top, static_cast<int>(width), static_cast<int>(height)};
}

std::vector<cv::Rect> DetectAndDraw::parse_detections(cv::Mat& output, const cv::Mat& image,
                                                      std::vector<int>& classIds, std::vector<float>& scores) const {
    int numDetections = output.size[2];
    int numFeatures = output.size[1];

    cv::Mat output2D = output.reshape(1, numFeatures).t();
    float x_factor = static_cast<float>(image.cols) / static_cast<float>(INP_WIDTH);
    float y_factor = static_cast<float>(image.rows) / static_cast<float>(INP_HEIGHT);

    std::vector<cv::Rect> boxes;
    std::vector<int> detectedClassIds;

    for (int i = 0; i < numDetections; ++i) {
        const float* data = output2D.ptr<float>(i);
        cv::Mat scoresMat(1, 3, CV_32FC1, const_cast<void*>(static_cast<const void*>(data + 4)));
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scoresMat, nullptr, &maxClassScore, nullptr, &classIdPoint);

        if (maxClassScore > CONF_THRESHOLD) {
            boxes.push_back(compute_bounding_box(data, x_factor, y_factor));
            scores.push_back(static_cast<float>(maxClassScore));
            detectedClassIds.push_back(classIdPoint.x);
        }
    }

    classIds = detectedClassIds;
    return boxes;
}

std::vector<cv::Rect> DetectAndDraw::detect_objects(const cv::Mat& image,
                                                    std::vector<int>& classIds,
                                                    std::vector<float>& confidences) {
    cv::Mat blob = preprocess_image(image);
    if (blob.empty()) return {};

    net.setInput(blob);
    cv::Mat output = net.forward();

    std::vector<float> scores;
    std::vector<cv::Rect> boxes = parse_detections(output, image, classIds, scores);

    // Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<cv::Rect> filteredBoxes;
    std::vector<int> filteredClassIds;
    std::vector<float> filteredConfidences;

    for (int idx : indices) {
        filteredBoxes.push_back(boxes[idx]);
        filteredClassIds.push_back(classIds[idx]);
        filteredConfidences.push_back(scores[idx]);
    }

    classIds = filteredClassIds;
    confidences = filteredConfidences;
    return filteredBoxes;
}

void DetectAndDraw::draw_detections(cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<int>& classIds) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 100), 2);

        std::string classname = (classIds[i] == 0) ? "Bear" : (classIds[i] == 1) ? "Frog" : "Cola";

        int baseLine;
        cv::Size labelSize = cv::getTextSize(classname, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseLine);
        int textX = boxes[i].x + (boxes[i].width - labelSize.width) / 2;
        int textY = std::max(boxes[i].y - 10, labelSize.height);

        cv::putText(image, classname, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    }
}

void DetectAndDraw::reset_counts() {
    std::lock_guard<std::mutex> lock(count_mutex);
    total_counts = {{0, 0}, {1, 0}, {2, 0}};
    actual_counts = {{0, 0}, {1, 0}, {2, 0}};
    std::cout << "[INFO] Reset Total counts!\n";
}
