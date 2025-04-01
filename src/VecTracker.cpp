#include "VecTracker.h"
#include "Detect_and_Draw.h"

VecTracker::VecTracker(const cv::Rect& box, int class_id)
    : last_box(box), current_box(box), class_id(class_id) {}

void VecTracker::update_box(const cv::Rect& new_box) {
    last_box = current_box;
    current_box = new_box;
}

void VecTracker::compute_direction_vectors(const std::vector<VecTracker>& all_objects) {
    direction_vectors.clear();
    cv::Point center = (current_box.tl() + current_box.br()) * 0.5;

    for (const auto& other : all_objects) {
        if (this == &other) continue;
        cv::Point other_center = (other.current_box.tl() + other.current_box.br()) * 0.5;
        direction_vectors.emplace_back(other_center - center);
    }

    std::ranges::sort(direction_vectors, [](const cv::Point& a, const cv::Point& b) {
        return std::tie(a.x, a.y) < std::tie(b.x, b.y);
    });
}

bool VecTracker::is_similar(const VecTracker& other, float tolerance) const {
    if (class_id != other.class_id) return false;
    if (direction_vectors.size() != other.direction_vectors.size()) return false;

    for (size_t i = 0; i < direction_vectors.size(); ++i) {
        auto dx = static_cast<float>(direction_vectors[i].x - other.direction_vectors[i].x);
        auto dy = static_cast<float>(direction_vectors[i].y - other.direction_vectors[i].y);
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist > tolerance) return false;
    }
    return true;
}

std::vector<VecTracker> VecTracker::update_trackers(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classIds,
    const std::vector<VecTracker>& previous_trackers,
    float tolerance
) {
    std::vector<VecTracker> current_trackers;
    for (size_t i = 0; i < boxes.size(); ++i) {
        current_trackers.emplace_back(boxes[i], classIds[i]);
    }

    for (auto& tracker : current_trackers)
        tracker.compute_direction_vectors(current_trackers);

    std::vector<bool> matched(current_trackers.size(), false);

    // Vergleich mit vorherigen Trackern
    for (const auto& prev : previous_trackers) {
        for (size_t i = 0; i < current_trackers.size(); ++i) {
            if (!matched[i] && current_trackers[i].is_similar(prev, tolerance)) {
                matched[i] = true;
                break;
            }
        }
    }

    // Nur zählen, wenn neue Tracker dazukommen
    int prev_size = static_cast<int>(previous_trackers.size());
    int curr_size = static_cast<int>(current_trackers.size());

    int new_objects = curr_size - prev_size;
    if (new_objects > 0) {
        int counted = 0;
        for (size_t i = 0; i < matched.size(); ++i) {
            if (!matched[i]) {
                int cid = current_trackers[i].get_class_id();
                total_counts[cid]++;
                counted++;
                if (counted >= new_objects) break; // nur genau so viele neue wie Differenz
            }
        }
    }

    return current_trackers;
}
