#ifndef CAMERA_MODE_H
#define CAMERA_MODE_H

#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include "SORT.h"  // Für den Tracker

// Globale Variablen für Kamera-Threads
extern std::mutex frame_mutex;
extern std::atomic<bool> stopThreads;
extern cv::Mat shared_frame;
extern SORT tracker;  // **Extern deklarieren, um den Tracker zu nutzen**

// Funktionen
void camera_capture(int camID);
void camera_processing(cv::dnn::Net& net);
void camera_thread(cv::dnn::Net& net, int camID);

#endif // CAMERA_MODE_H
