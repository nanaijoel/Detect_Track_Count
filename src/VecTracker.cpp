#include "VecTracker.h"
#include <cmath>
#include <algorithm>
#include <iostream>

int VecTracker::next_id = 0;

VecTracker::VecTracker(const cv::Rect& bbox)
    : bounding_box(bbox), id(next_id++) {}

cv::Point VecTracker::get_center() const {
    return {
        bounding_box.x + bounding_box.width / 2,
        bounding_box.y + bounding_box.height / 2
    };
}

cv::Rect VecTracker::get_bounding_box() const {
    return bounding_box;
}

void VecTracker::compute_relative_distances(const std::vector<cv::Rect>& all_bboxes) {
    relative_distances.clear();
    cv::Point my_center = get_center();

    for (const auto& other_bbox : all_bboxes) {
        if (other_bbox == bounding_box) continue;

        cv::Point other_center = {
            other_bbox.x + other_bbox.width / 2,
            other_bbox.y + other_bbox.height / 2
        };

        float dx = static_cast<float>(other_center.x - my_center.x);
        float dy = static_cast<float>(other_center.y - my_center.y);
        float dist = std::sqrt(dx * dx + dy * dy);
        relative_distances.push_back(dist);
    }

    std::ranges::sort(relative_distances);
}

const std::vector<float>& VecTracker::get_relative_distances() const {
    return relative_distances;
}

int VecTracker::get_id() const {
    return id;
}

void VecTracker::set_id(int new_id) {
    id = new_id;
}

int VecTracker::get_class_id() const {
    return class_id;
}

void VecTracker::set_class_id(int cid) {
    class_id = cid;
}

int VecTracker::get_age() const {
    return age;
}

void VecTracker::increment_age() {
    age++;
}

void VecTracker::reset_age() {
    age = 0;
}

bool VecTracker::is_potential_match(const VecTracker& other, float tolerance) const {
    const auto& other_distances = other.get_relative_distances();
    if (relative_distances.size() != other_distances.size()) return false;

    for (size_t i = 0; i < relative_distances.size(); ++i) {
        if (std::abs(relative_distances[i] - other_distances[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<VecTracker> VecTracker::update_trackers(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classIds,
    std::vector<VecTracker>& previous_trackers
) {
    std::vector<std::pair<cv::Rect, int>> box_class_pairs;
    for (size_t i = 0; i < boxes.size(); ++i) {
        box_class_pairs.emplace_back(boxes[i], classIds[i]);
    }

    std::sort(box_class_pairs.begin(), box_class_pairs.end(), [](const auto& a, const auto& b) {
        int ay = a.first.y + a.first.height / 2;
        int by = b.first.y + b.first.height / 2;
        if (ay != by) return ay < by;
        int ax = a.first.x + a.first.width / 2;
        int bx = b.first.x + b.first.width / 2;
        return ax < bx;
    });

    std::vector<VecTracker> current_trackers;
    std::vector<cv::Rect> sorted_boxes;
    std::vector<int> sorted_classIds;

    for (const auto& pair : box_class_pairs) {
        VecTracker tracker(pair.first);
        tracker.set_class_id(pair.second);
        current_trackers.push_back(tracker);
        sorted_boxes.push_back(pair.first);
        sorted_classIds.push_back(pair.second);
    }

    for (auto& tracker : current_trackers) {
        tracker.compute_relative_distances(sorted_boxes);
    }

    std::vector<int> assigned_ids(current_trackers.size(), -1);
    std::vector<bool> prev_assigned(previous_trackers.size(), false);

    for (size_t i = 0; i < current_trackers.size(); ++i) {
        const auto& curr_distances = current_trackers[i].get_relative_distances();
        float best_score = std::numeric_limits<float>::max();
        int best_match_idx = -1;

        for (size_t j = 0; j < previous_trackers.size(); ++j) {
            if (prev_assigned[j]) continue;
            const auto& prev_distances = previous_trackers[j].get_relative_distances();
            if (curr_distances.size() != prev_distances.size()) continue;

            float score = 0.0f;
            for (size_t k = 0; k < curr_distances.size(); ++k) {
                float diff = curr_distances[k] - prev_distances[k];
                score += diff * diff;
            }

            if (score < best_score) {
                best_score = score;
                best_match_idx = static_cast<int>(j);
            }
        }

        if (best_match_idx != -1) {
            current_trackers[i].set_id(previous_trackers[best_match_idx].get_id());
            current_trackers[i].set_class_id(previous_trackers[best_match_idx].get_class_id());
            assigned_ids[i] = previous_trackers[best_match_idx].get_id();
            prev_assigned[best_match_idx] = true;
        }

        std::cout << "Tracker " << i << " matched with score: " << best_score
                  << " and class_id: " << current_trackers[i].get_class_id()
                  << std::endl;



    }

    previous_trackers = current_trackers;
    return current_trackers;
}
