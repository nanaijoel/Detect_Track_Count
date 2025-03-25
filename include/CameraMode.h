#ifndef CAMERA_MODE_H
#define CAMERA_MODE_H

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include "Detect_and_Draw.h"
#include "Sort.h"

class ObjectDetectionGUI;
class QApplication;


extern SORT tracker;

void camera_thread(DetectAndDraw& detector, int camID, QApplication& app);

void camera_capture(int camID);

void camera_processing(DetectAndDraw& detector, ObjectDetectionGUI* gui);

#endif // CAMERA_MODE_H
