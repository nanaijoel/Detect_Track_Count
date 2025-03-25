#include "Detect_and_Draw.h"
#include "ImageMode.h"
#include "CameraMode.h"
#include <iostream>
#include <windows.h>
#include <QApplication>
#include <Eigen/Dense>


int main(int argc, char *argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    QApplication app(argc, argv);
    DetectAndDraw detector("../best.onnx");

    int mode;
    std::cout << "Type 1 or 2 in console and press ENTER:\n"
                 "1 for Image-Detection, 2 for Live-Camera Tracking): ";
    std::cin >> mode;

    if (mode == 1) {
        image_mode(detector, "../images/");
        return 0;
    } else if (mode == 2) {
        camera_thread(detector, 0, app);
        return 0;
    }

    return 0;
}
