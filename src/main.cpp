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
    std::cout << "Wähle Modus: (1) Image-Detection, (2) Live-Camera): ";
    std::cin >> mode;

    if (mode == 1) {
        image_mode(net, "../images/");
    }
    else if (mode == 2) {
        camera_thread(net, 0);
    }

    return 0;
}
