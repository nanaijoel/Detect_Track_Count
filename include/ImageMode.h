#ifndef IMAGE_MODE_H
#define IMAGE_MODE_H

#include <opencv2/dnn.hpp>
#include <string>

void image_mode(cv::dnn::Net& net, const std::string& imgDirectory);

#endif // IMAGE_MODE_H
