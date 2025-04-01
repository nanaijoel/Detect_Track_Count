#include "CameraMode.h"
#include "Detect_and_Draw.h"
#include <thread>
#include <QApplication>
#include "GUI.h"
#include "TotalCounter.h"
#include "BYTETracker.h"
#include "STrack.h"
#include "Rect.h"
#include "Object.h"

using namespace byte_track;


std::mutex frame_mutex;
std::atomic<bool> stopThreads(false);
TotalCounter totalCounter;
std::map<int, int> actual_counts = {{0, 0}, {1, 0}, {2, 0}};

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
    BYTETracker tracker(30, 0.6f);  // FPS, matching threshold

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

        DetectAndDraw::draw_detections(frame, boxes, classIds);

        // === Prepare detections for ByteTrack ===
        std::vector<Object> detections;
        for (size_t i = 0; i < boxes.size(); ++i) {
            const cv::Rect2f r = boxes[i];
            Rect rect(r.x, r.y, r.width, r.height);
            detections.emplace_back(rect, classIds[i], confidences[i]);
        }

        std::vector<byte_track::BYTETracker::STrackPtr> tracks = tracker.update(detections);

        // === DEBUG: Output number of active ByteTrack tracks ===
        std::cout << "[ByteTrack] Active tracks: " << tracks.size() << std::endl;

        {
            std::lock_guard<std::mutex> lock(count_mutex);
            actual_counts.clear();
            for (int id : classIds)
                actual_counts[id]++;
        }

        // === GUI-Update ===
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
