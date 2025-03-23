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
    // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // 0.25 = manually
    // cap.set(cv::CAP_PROP_EXPOSURE, -3.9);         //  -2 to -6
    // cap.set(cv::CAP_PROP_BRIGHTNESS, 155);      // values (50–255)
    // cap.set(cv::CAP_PROP_CONTRAST, 90);         // 0-100

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

    cv::namedWindow("Live YOLO SORT Detection");
    cv::setMouseCallback("Live YOLO SORT Detection", DetectAndDraw::mouse_callback, nullptr);
    while (!stopThreads) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (shared_frame.empty()) continue;
            frame = shared_frame.clone();
        }

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes = detector.detect_objects(frame, classIds, confidences);
        int frame_width = frame.cols;

        tracker.update_tracks(boxes, classIds, confidences, frame_width);

        DetectAndDraw::draw_detections(frame, boxes, classIds);

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            actual_counts.clear();

            for (int classId : classIds) {
                actual_counts[classId]++;
            }
        }

        int scanline_x = frame.cols / 2;
        cv::line(frame, cv::Point(scanline_x, 0), cv::Point(scanline_x, frame.rows), cv::Scalar(0, 255, 255), 2);
        cv::Mat info_panel = DetectAndDraw::create_info_panel(frame.rows);
        cv::Mat combined_output;
        cv::hconcat(frame, info_panel, combined_output);
        cv::imshow("Live YOLO SORT Detection", combined_output);
        cv::setMouseCallback("Live YOLO SORT Detection", DetectAndDraw::mouse_callback, nullptr);

        if (cv::waitKey(1) == 27) stopThreads = true;
    }
}


void camera_thread(DetectAndDraw& detector, int camID) {
    std::thread captureThread(camera_capture, camID);

    std::thread processingThread([&]() { camera_processing(detector); });

    captureThread.join();
    processingThread.join();
}
