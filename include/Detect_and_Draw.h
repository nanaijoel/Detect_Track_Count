#ifndef DETECT_AND_DRAW_H
#define DETECT_AND_DRAW_H

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


constexpr float CONF_THRESHOLD = 0.70f;
constexpr float NMS_THRESHOLD = 0.45f;
constexpr int INP_WIDTH = 640, INP_HEIGHT = 640;


class DetectAndDraw {
public:
    explicit DetectAndDraw(const std::string& model_path);

    std::vector<cv::Rect> detect_objects(const cv::Mat& image, std::vector<int>& classIds);

    static void mouse_callback(int event, int x, int y, int flags, void* userdata);
    static void reset_counts();
    static void draw_detections(cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<int>& classIds);
    static cv::Mat create_info_panel(int height);
    static cv::Mat preprocess_image(const cv::Mat& image);
    static cv::Rect compute_bounding_box(const float* data, float x_factor, float y_factor);

private:
    cv::dnn::Net net;

    static std::vector<cv::Rect> parse_detections(cv::Mat& output, const cv::Mat& image, std::vector<int>& classIds, std::vector<float>& scores);
};

#endif // DETECT_AND_DRAW_H
