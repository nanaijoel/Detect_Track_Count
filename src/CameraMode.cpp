#include "CameraMode.h"
#include "SORT.h"
#include "Detect_and_Draw.h"
#include <opencv2/opencv.hpp>

std::mutex frame_mutex;
cv::Mat shared_frame;

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Couldn't open camera frame - Error!\n";
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
    cv::namedWindow("Live YOLO SORT Detection", cv::WINDOW_NORMAL);

    while (!stopThreads) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (shared_frame.empty() || shared_frame.rows == 0 || shared_frame.cols == 0) continue;
            frame = shared_frame.clone();
        }

        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = detect_objects(net, frame, classIds);

        tracker.update_tracks(boxes, classIds);
        draw_detections(frame, boxes, classIds);

        std::map<int, int> temp_actual_counts;
        for (int id : classIds) {
            temp_actual_counts[id]++;
        }

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            actual_counts = temp_actual_counts;
        }

        cv::Mat info_panel = create_info_panel(frame.rows);
        if (!info_panel.empty() && info_panel.rows > 0 && info_panel.cols > 0)
            cv::resize(info_panel, info_panel, cv::Size(frame.cols / 3, frame.rows));

        cv::Mat combined_output;
        cv::hconcat(frame, info_panel, combined_output);

        cv::resizeWindow("Live YOLO SORT Detection", combined_output.cols, combined_output.rows);
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
