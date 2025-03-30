#ifndef CAMERA_MODE_H
#define CAMERA_MODE_H

#include "Detect_and_Draw.h"

class ObjectDetectionGUI;
class QApplication;


void camera_thread(DetectAndDraw& detector, int camID, QApplication& app);

void camera_capture(int camID);

void camera_processing(DetectAndDraw& detector, ObjectDetectionGUI* gui);

#endif // CAMERA_MODE_H
