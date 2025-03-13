#include "Detect_and_Draw.h"
#include "ImageMode.h"
#include "CameraMode.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

namespace fs = std::filesystem;


int main() {
    SetConsoleOutputCP(CP_UTF8);

    cv::dnn::Net net = load_model("../best.onnx");

    int mode;
    std::cout << "Type in console 1 or 2 in console and press ENTER:\n"
                 "'1' for Image-Detection, '2' for Live-Camera Tracking): ";
    std::cin >> mode;

    if (mode == 1) {
        image_mode(net, "../images/");
    }
    else if (mode == 2) {
        camera_thread(net, 0);
    }

    return 0;
}
