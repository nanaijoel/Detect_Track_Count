#include "CameraMode.h"
#include "SORT.h"
#include "Detect_and_Draw.h"
#include <opencv2/opencv.hpp>

std::mutex frame_mutex;
cv::Mat shared_frame;

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    if (!cap.isOpened()) {
        std::cerr << "Fehler beim Öffnen der Kamera!\n";
        return;
    }

    while (!stopThreads) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        std::lock_guard<std::mutex> lock(frame_mutex);
        shared_frame = frame.clone();
    }
    cap.release();
}

void camera_processing(cv::dnn::Net& net) {
    while (!stopThreads) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (shared_frame.empty()) continue;
            frame = shared_frame.clone();
        }

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detect_objects(net, frame, classIds);

        tracker.update_tracks(boxes, classIds);
        draw_detections(frame, boxes, classIds);

        // **Neues Info-Panel abrufen**
        cv::Mat info_panel = create_info_panel(frame.rows);

        // **Kamera-Output + Info-Panel zusammenfügen**
        cv::Mat combined_output;
        cv::hconcat(frame, info_panel, combined_output);
        cv::imshow("Live YOLO SORT Detection", combined_output);

        if (cv::waitKey(1) == 27) stopThreads = true;
    }
}

void camera_thread(cv::dnn::Net& net, int camID) {
    std::thread captureThread(camera_capture, camID);
    std::thread processingThread(camera_processing, std::ref(net));
    captureThread.join();
    processingThread.join();
}
