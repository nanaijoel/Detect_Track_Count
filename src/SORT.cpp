#include "Sort.h"
#include <iostream>
#include "TrackRecoveryHelper.h"

std::mutex count_mutex;
std::atomic<bool> stopThreads(false);
std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};
std::map<int, int> actual_counts = {{0, 0}, {1, 0}, {2, 0}};

SORT tracker;
TrackRecoveryHelper recovery_helper;

SORT::Track::Track(int track_id, cv::Rect bbox, int class_id) {
    id = track_id;
    box = bbox;
    classId = class_id;
    last_confidence = 0.0f;
    stable_class = -1;
    frames_since_seen = 0;
    matched_in_this_frame = false;
    was_counted = false;
    crossed_scanline = false;

    kf = cv::KalmanFilter(4, 2, 0);
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                     0, 1, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);
    kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    kf.statePost.at<float>(0) = static_cast<float>(bbox.x) + static_cast<float>(bbox.width) / 2.0f;
    kf.statePost.at<float>(1) = static_cast<float>(bbox.y) + static_cast<float>(bbox.height) / 2.0f;

}

void SORT::Track::predict() {
    cv::Mat pred = kf.predict();
    box.x = static_cast<int>(std::round(pred.at<float>(0) - static_cast<float>(box.width) / 2.f));
    box.y = static_cast<int>(std::round(pred.at<float>(1) - static_cast<float>(box.height) / 2.f));
    frames_since_seen++;
    matched_in_this_frame = false;
}

void SORT::Track::update(cv::Rect new_box, int new_class, float conf) {
    cv::Mat meas = (cv::Mat_<float>(2, 1) <<
        static_cast<float>(new_box.x) + static_cast<float>(new_box.width) / 2.0f,
        static_cast<float>(new_box.y) + static_cast<float>(new_box.height) / 2.0f);

    kf.correct(meas);
    box = new_box;
    frames_since_seen = 0;
    matched_in_this_frame = true;

    last_confidence = conf;
    class_confidences[new_class].push_back(conf);

    if (class_confidences[new_class].size() >= 5) {
        stable_class = new_class;
    }
}


void SORT::match_existing_tracks(const std::vector<cv::Rect>& detected_boxes,
                                 const std::vector<int>& classIds,
                                 const std::vector<float>& confidences,
                                 std::vector<bool>& matched) {
    for (auto& track : tracks) {
        float best_iou = 0;
        int best_match = -1;

        for (int i = 0; i < static_cast<int>(detected_boxes.size()); i++) {
            if (track.classId != classIds[i]) continue;

            float intersection_area = static_cast<float>((track.box & detected_boxes[i]).area());
            float union_area = static_cast<float>(track.box.area()) + static_cast<float>(detected_boxes[i].area()) - intersection_area;
            float iou = intersection_area / union_area;

            float dist = std::hypot(
                static_cast<float>(track.box.x - detected_boxes[i].x),
                static_cast<float>(track.box.y - detected_boxes[i].y)
            );

            if ((iou > best_iou && iou > 0.4f) || (iou > 0.2f && dist < 35.0f)) {
                best_iou = iou;
                best_match = i;
            }
        }

        if (best_match != -1) {
            track.update(detected_boxes[best_match], classIds[best_match], confidences[best_match]);
            matched[best_match] = true;
        } else {
            recovery_helper.store_unmatched_track(track);
        }
    }
}


void SORT::add_new_tracks(const std::vector<cv::Rect>& detected_boxes,
                          const std::vector<int>& classIds,
                          const std::vector<float>& confidences,
                          std::vector<bool>& matched) {
    for (size_t i = 0; i < detected_boxes.size(); i++) {
        if (!matched[i]) {
            Track new_track(next_id++, detected_boxes[i], classIds[i]);
            new_track.last_confidence = confidences[i];
            tracks.push_back(new_track);
        }
    }
}


void SORT::remove_old_tracks() {
    for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->frames_since_seen > 10) {
            it = tracks.erase(it);
        } else {
            ++it;
        }
    }
}

void SORT::update_counts(int scanline_x) {
    std::lock_guard<std::mutex> lock(count_mutex);
    actual_counts.clear();

    for (auto& track : tracks) {
        actual_counts[track.classId]++;

        int left_x = track.box.x;
        int right_x = track.box.x + track.box.width;

        bool is_crossing = (left_x <= scanline_x && right_x >= scanline_x);

        if (is_crossing && !track.crossed_scanline) {
            total_counts[track.classId]++;
            track.crossed_scanline = true;
        }

        if (!is_crossing) {
            track.crossed_scanline = false;
        }
    }
}

void SORT::update_tracks(const std::vector<cv::Rect>& detected_boxes,
                         const std::vector<int>& classIds,
                         const std::vector<float>& confidences,
                         int frame_width) {
    std::vector<bool> matched(detected_boxes.size(), false);

    for (auto& track : tracks) {
        track.predict();
    }

    match_existing_tracks(detected_boxes, classIds, confidences, matched);
    recovery_helper.update_missing();
    recovery_helper.try_recover(detected_boxes, classIds, matched, tracks);
    add_new_tracks(detected_boxes, classIds, confidences, matched);
    remove_old_tracks();
    update_counts(frame_width / 2);
}


std::vector<SORT::Track> SORT::get_tracks() const {
    return tracks;
}
