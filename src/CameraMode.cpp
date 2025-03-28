#include "CameraMode.h"
#include "Detect_and_Draw.h"  // shared_frame ist hier 'extern' deklariert
#include <thread>
#include <QApplication>
#include "GUI.h"

std::mutex frame_mutex;

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
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

void camera_processing(DetectAndDraw& detector, ObjectDetectionGUI* gui) {
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

        if (gui) {
            QMetaObject::invokeMethod(gui, [frame, gui]() {
                gui->updateFrame(frame);
            }, Qt::QueuedConnection);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void camera_thread(DetectAndDraw& detector, int camID, QApplication& app) {
    ObjectDetectionGUI gui(&detector);
    gui.setWindowTitle("YOLO SORT LIVE GUI");
    gui.showMaximized();

    std::thread captureThread(camera_capture, camID);
    std::thread processingThread([&]() { camera_processing(detector, &gui); });

    QApplication::exec();
    stopThreads = true;

    captureThread.join();
    processingThread.join();
}
