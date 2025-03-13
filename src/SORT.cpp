#include "Sort.h"

std::mutex count_mutex;
std::atomic<bool> stopThreads(false);
std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};
std::map<int, int> actual_counts = {{0, 0}, {1, 0}, {2, 0}};

SORT tracker;

SORT::Track::Track(int track_id, cv::Rect bbox, int class_id) {
    id = track_id;
    box = bbox;
    classId = class_id;
    frames_since_seen = 0;
    matched_in_this_frame = false;

    kf = cv::KalmanFilter(4, 2, 0);
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                     0, 1, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);
    kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    kf.statePost.at<float>(0) = static_cast<float>(bbox.x) + static_cast<float>(bbox.width) / 2.f;
    kf.statePost.at<float>(1) = static_cast<float>(bbox.y) + static_cast<float>(bbox.height) / 2.f;
}


void SORT::Track::update(cv::Rect new_box) {
    cv::Mat meas = (cv::Mat_<float>(2, 1) << new_box.x + new_box.width / 2, new_box.y + new_box.height / 2);
    kf.correct(meas);
    box = new_box;
    frames_since_seen = 0;
    matched_in_this_frame = true;
}


void SORT::Track::predict() {
    cv::Mat pred = kf.predict();
    box.x = static_cast<int>(std::round(pred.at<float>(0) - static_cast<float>(box.width) / 2.f));
    box.y = static_cast<int>(std::round(pred.at<float>(1) - static_cast<float>(box.height) / 2.f));
    frames_since_seen++;
    matched_in_this_frame = false;
}


void SORT::match_existing_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds, std::vector<bool>& matched) {
    for (auto& track : tracks) {
        float best_iou = 0;
        int best_match = -1;

        for (int i = 0; i < detected_boxes.size(); i++) {
            float intersection_area = static_cast<float>((track.box & detected_boxes[i]).area());
            float union_area = static_cast<float>(track.box.area()) + static_cast<float>(detected_boxes[i].area()) - intersection_area;
            float iou = intersection_area / union_area;

            float dist = std::sqrt(std::pow(static_cast<float>(track.box.x - detected_boxes[i].x), 2.f) +
                std::pow(static_cast<float>(track.box.y - detected_boxes[i].y), 2.f));


            if ((iou > best_iou && iou > 0.4) || (iou > 0.2 && dist < 35)) {
                best_iou = iou;
                best_match = i;
            }
        }

        if (best_match != -1) {
            track.update(detected_boxes[best_match]);
            matched[best_match] = true;
        }
    }
}


void SORT::add_new_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds, std::vector<bool>& matched) {
    for (size_t i = 0; i < detected_boxes.size(); i++) {
        if (!matched[i]) {
            bool is_truly_new = true;

            for (const auto& track : tracks) {
                float dist = std::sqrt(std::pow(static_cast<float>(track.box.x - detected_boxes[i].x), 2.f) +
                       std::pow(static_cast<float>(track.box.y - detected_boxes[i].y), 2.f));

                if (dist < 50 && track.frames_since_seen < 3) {
                    is_truly_new = false;
                    break;
                }
            }

            tracks.emplace_back(next_id++, detected_boxes[i], classIds[i]);

            if (is_truly_new) {
                std::lock_guard<std::mutex> lock(count_mutex);
                total_counts[classIds[i]]++;
            }
        }
    }
}


void SORT::remove_old_tracks() {
    std::erase_if(tracks, [](const Track& t) { return t.frames_since_seen > 30; });
}


void SORT::update_counts() {
    std::lock_guard<std::mutex> lock(count_mutex);

    actual_counts.clear();
    for (const auto& track : tracks) {
        actual_counts[track.classId]++;
    }
}


void SORT::update_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds) {
    std::vector<bool> matched(detected_boxes.size(), false);

    for (auto& track : tracks) {
        track.predict();
    }

    match_existing_tracks(detected_boxes, classIds, matched);
    add_new_tracks(detected_boxes, classIds, matched);
    remove_old_tracks();
    update_counts();
}

std::vector<SORT::Track> SORT::get_tracks() const {
    return tracks;
}

