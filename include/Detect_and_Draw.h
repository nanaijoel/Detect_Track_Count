#ifndef DETECT_AND_DRAW_H
#define DETECT_AND_DRAW_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <map>
#include <mutex>
#include <vector>
#include <string>


extern std::map<int, int> total_counts;
extern std::map<int, int> actual_counts;
extern std::mutex count_mutex;
extern cv::Mat shared_frame;

class DetectAndDraw {
public:
    explicit DetectAndDraw(const std::string& model_path);

    std::vector<cv::Rect> detect_objects(const cv::Mat& image,
                                         std::vector<int>& classIds,
                                         std::vector<float>& confidences);

    static void draw_detections(cv::Mat& image,
                                const std::vector<cv::Rect>& boxes,
                                const std::vector<int>& classIds);

    static void reset_counts();

private:
    cv::dnn::Net net;

    cv::Mat preprocess_image(const cv::Mat& image) const;

    std::vector<cv::Rect> parse_detections(cv::Mat& output,
                                           const cv::Mat& image,
                                           std::vector<int>& classIds,
                                           std::vector<float>& scores) const;

    static cv::Rect compute_bounding_box(const float* data,
                                         float x_factor,
                                         float y_factor);

    const float CONF_THRESHOLD = 0.25f;
    const float NMS_THRESHOLD = 0.45f;
    const int INP_WIDTH = 640;
    const int INP_HEIGHT = 640;
};

#endif // DETECT_AND_DRAW_H
