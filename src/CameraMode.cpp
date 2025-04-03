#include "CameraMode.h"
#include "Detect_and_Draw.h"
#include <thread>
#include <QApplication>
#include "GUI.h"
#include "TotalCounter.h"
#include "BYTETracker.h"
#include "Rect.h"
#include "Object.h"
#include <chrono>

using namespace byte_track;


std::mutex frame_mutex;
std::atomic<bool> stopThreads(false);
TotalCounter totalCounter;
std::map<int, int> actual_counts = {{0, 0}, {1, 0}, {2, 0}};

void camera_capture(int camID) {
    cv::VideoCapture cap(camID);
    //cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1024);
    //cap.set(cv::CAP_PROP_FPS, 30);
    // std::cout << "[INFO] CAP_PROP_FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    // std::cout << "[INFO] FOURCC: " << int(cap.get(cv::CAP_PROP_FOURCC)) << std::endl;

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

        // in camera_processing():
        // static int frame_count = 0;
        // static auto last_time = std::chrono::steady_clock::now();
        //
        // frame_count++;
        // auto now = std::chrono::steady_clock::now();
        // double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        //
        // if (elapsed >= 1000.0) {
        //     double fps = frame_count * 1000.0 / elapsed;
        //     std::cout << "[INFO] Measured FPS: " << fps << std::endl;
        //
        //     frame_count = 0;
        //     last_time = now;
        // }

    }
    cap.release();
}

void camera_processing(DetectAndDraw& detector, ObjectDetectionGUI* gui) {
    BYTETracker tracker(10, 30);

    while (!stopThreads) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (shared_frame.empty()) continue;
            frame = shared_frame.clone();
        }

        std::vector<int> classIds;
        std::vector<float> confidences;

        // static int frame_count = 0;
        // static auto last_time = std::chrono::steady_clock::now();
        //
        // frame_count++;
        // auto now = std::chrono::steady_clock::now();

        std::vector<cv::Rect> boxes = detector.detect_objects(frame, classIds, confidences);

        DetectAndDraw::draw_detections(frame, boxes, classIds);

        // double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        // std::cout << "Vergangene Zeit für CNN: " << elapsed << std::endl;
        // frame_count = 0;
        // last_time = now;

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
        totalCounter.update(tracks);

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
