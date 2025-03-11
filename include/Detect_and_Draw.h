#ifndef DETECT_AND_DRAW_H
#define DETECT_AND_DRAW_H

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>


extern std::map<int, int> actual_counts;
extern std::map<int, int> total_counts;

const float CONF_THRESHOLD = 0.70f;
const float NMS_THRESHOLD = 0.45f;
const int INP_WIDTH = 640, INP_HEIGHT = 640;

cv::Mat preprocess_image(const cv::Mat& image);
cv::dnn::Net load_model(const std::string& model_path);
std::vector<cv::Rect> detect_objects(cv::dnn::Net& net, const cv::Mat& image, std::vector<int>& classIds);
void draw_detections(cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<int>& classIds);
cv::Mat create_info_panel(int height);

#endif // DETECT_AND_DRAW_H
