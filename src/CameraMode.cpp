#include "CameraMode.h"
#include <thread>


std::mutex frame_mutex;
cv::Mat shared_frame;

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    if (!cap.isOpened()) {
        std::cerr << "Couldn't open camera - Error!\n";
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

void camera_processing(DetectAndDraw& detector) {
    while (!stopThreads) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (shared_frame.empty()) continue;
            frame = shared_frame.clone();
        }

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detector.detect_objects(frame, classIds);

        tracker.update_tracks(boxes, classIds);
        DetectAndDraw::draw_detections(frame, boxes, classIds);

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            actual_counts.clear();

            for (int classId : classIds) {
                actual_counts[classId]++;
            }
        }

        cv::Mat info_panel = DetectAndDraw::create_info_panel(frame.rows);
        cv::Mat combined_output;
        cv::hconcat(frame, info_panel, combined_output);
        cv::imshow("Live YOLO SORT Detection", combined_output);

        if (cv::waitKey(1) == 27) stopThreads = true;
    }
}


void camera_thread(DetectAndDraw& detector, int camID) {
    std::thread captureThread(camera_capture, camID);

    std::thread processingThread([&]() { camera_processing(detector); });

    captureThread.join();
    processingThread.join();
}
