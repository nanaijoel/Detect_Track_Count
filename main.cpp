//
// Created by j.nanai on 09.03.2025.
//
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <windows.h>
#include <filesystem>
#include <set>
#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

// Hyperparameter
const float CONF_THRESHOLD = 0.65f;
const float NMS_THRESHOLD = 0.45f;
const int INP_WIDTH = 640, INP_HEIGHT = 640;

// **Comparator für std::set<cv::Point>**
struct PointComparator {
    bool operator()(const cv::Point& p1, const cv::Point& p2) const {
        return (p1.x < p2.x) || (p1.x == p2.x && p1.y < p2.y);
    }
};

// **Globale Variablen für Tracking & Multithreading**
std::set<cv::Point, PointComparator> trackedObjects;
std::mutex count_mutex;
std::atomic<bool> stopThreads(false);
std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}}; // Bear, Frog, Cola

// **YOLO-Modell laden**
cv::dnn::Net load_model(const std::string& model_path) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Fehler beim Laden des Modells!\n";
        exit(-1);
    }
    return net;
}

// **Bild vorbereiten**
cv::Mat preprocess_image(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INP_WIDTH, INP_HEIGHT), cv::Scalar(0, 0, 0), true, false);
    return blob;
}

// **Objekterkennung mit YOLO + NMS**
std::vector<cv::Rect> detect_objects(cv::dnn::Net& net, const cv::Mat& image, std::vector<int>& classIds) {
    cv::Mat blob = preprocess_image(image);
    net.setInput(blob);
    cv::Mat output = net.forward();

    int numDetections = output.size[2];
    int numFeatures = output.size[1];

    if (numFeatures != 4 + 3) {
        std::cerr << "Fehler: Falsche Anzahl an Features erkannt.\n";
        return {};
    }

    cv::Mat output2D = output.reshape(1, numFeatures).t();
    float x_factor = (float)image.cols / (float)INP_WIDTH;
    float y_factor = (float)image.rows / (float)INP_HEIGHT;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

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
            int left = std::max(0, (int)(centerX - width / 2));
            int top = std::max(0, (int)(centerY - height / 2));

            boxes.emplace_back(left, top, (int)width, (int)height);
            scores.push_back((float)maxClassScore);
            classIds.push_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<cv::Rect> filteredBoxes;
    std::vector<int> filteredClassIds;

    for (int idx : indices) {
        filteredBoxes.push_back(boxes[idx]);
        filteredClassIds.push_back(classIds[idx]);
    }

    classIds = filteredClassIds;
    return filteredBoxes;
}

// **Bounding Boxes & Labels zeichnen**
void draw_detections(cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<int>& classIds) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 200), 3);

        std::string classname;
        switch (classIds[i]) {
            case 0: classname = "Bear"; break;
            case 1: classname = "Frog"; break;
            case 2: classname = "Cola"; break;
            default: classname = "Unknown"; break;
        }
        int baseLine;
        cv::Size labelSize = cv::getTextSize(classname, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseLine);
        int textX = boxes[i].x + (boxes[i].width - labelSize.width) / 2;
        int textY = std::max(boxes[i].y - 20, labelSize.height);
        cv::putText(image, classname, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 3);
    }
}

// **Image-Mode (mit Resizing)**
void image_mode(cv::dnn::Net& net, const std::string& imgDirectory) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(imgDirectory)) {
        std::string filePath = entry.path().string();
        if (filePath.find(".jpg") != std::string::npos) {
            imagePaths.push_back(filePath);
        }
    }

    if (imagePaths.empty()) {
        std::cerr << "Keine Bilder gefunden im Verzeichnis: " << imgDirectory << "\n";
        return;
    }

    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detect_objects(net, image, classIds);
        draw_detections(image, boxes, classIds);

        cv::resize(image, image, cv::Size(800, 600));
        cv::imshow("YOLO Image Detection", image);

        std::cout << "Drücke eine Taste für das nächste Bild oder ESC zum Beenden\n";
        int key = cv::waitKey(0);
        if (key == 27) break;
    }

    cv::destroyAllWindows();
}

// **Live-Kamera-Modus mit Count-Ausgabe**
void camera_thread(cv::dnn::Net& net, int camID) {
    cv::VideoCapture cap(camID);
    if (!cap.isOpened()) {
        std::cerr << "Fehler beim Öffnen der Kamera!\n";
        return;
    }

    while (!stopThreads) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detect_objects(net, frame, classIds);
        draw_detections(frame, boxes, classIds);

        std::map<int, int> actual_counts = {{0, 0}, {1, 0}, {2, 0}};

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            for (int classId : classIds) {
                actual_counts[classId]++;
                total_counts[classId]++;
            }
        }

        cv::Mat info_panel(frame.rows, 300, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::putText(info_panel, "OBJECT COUNTS",
                    cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 55, 255), 3);

        std::string classname;
        for (int i = 0; i < 3; i++) {

            switch (i) {
                case 0: classname = "Bear"; break;
                case 1: classname = "Frog"; break;
                case 2: classname = "Cola"; break;
                default: classname = "Unknown"; break;
            }

            cv::putText(info_panel, classname + " Actual: " + std::to_string(actual_counts[i]) +
                                      " | Total: " + std::to_string(total_counts[i]),
                        cv::Point(20, 100 + i * 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }

        cv::Mat combined_output;
        cv::hconcat(frame, info_panel, combined_output);
        cv::imshow("Live YOLO Detection", combined_output);

        if (cv::waitKey(1) == 27) stopThreads = true;
    }
    cap.release();
}

// **MAIN**
int main() {
    SetConsoleOutputCP(CP_UTF8);
    cv::dnn::Net net = load_model("../best.onnx");
    int mode;
    std::cout << "Wähle Modus: (1) Image-Detection, (2) Live-Camera: ";
    std::cin >> mode;

    if (mode == 1) image_mode(net, "../images/");
    else if (mode == 2) {
        std::thread camThread(camera_thread, std::ref(net), 0);
        camThread.join();
    }

    return 0;
}
