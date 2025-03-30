#include "CameraMode.h"
#include "Detect_and_Draw.h"
#include <thread>
#include <QApplication>
#include "GUI.h"
#include "VecTracker.h"

std::mutex frame_mutex;
std::atomic<bool> stopThreads(false);

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1024);
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
    std::vector<VecTracker> previous_trackers;

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

        // <<< NEU: zentraler Tracking-Aufruf >>>
        std::vector<VecTracker> current_trackers = VecTracker::update_trackers(boxes, classIds, previous_trackers);

        // <<< extrahieren der Ergebnisse für die GUI >>>
        std::vector<cv::Rect> sorted_boxes;
        std::vector<int> sorted_classIds;
        for (const auto& t : current_trackers) {
            sorted_boxes.push_back(t.get_bounding_box());
            sorted_classIds.push_back(t.get_class_id());
        }

        DetectAndDraw::draw_detections(frame, sorted_boxes, sorted_classIds);

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            actual_counts.clear();
            for (int classId : sorted_classIds) {
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