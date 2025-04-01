#include "TotalCounter.h"

std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};

void TotalCounter::update(const std::vector<int>& current_classIds) {
    std::map<int, int> current_counts;
    for (int id : current_classIds) {
        current_counts[id]++;
    }

    for (const auto& [class_id, current_count] : current_counts) {
        int previous_count = previous_counts[class_id];

        if (current_count > previous_count) {
            frame_stability[class_id]++;
            if (frame_stability[class_id] >= stability_threshold) {
                int delta = current_count - previous_count;
                total_counts[class_id] += delta;
                frame_stability[class_id] = 0;
            }
        } else {
            frame_stability[class_id] = 0;
        }
    }

    for (auto it = previous_counts.begin(); it != previous_counts.end(); ) {
        int class_id = it->first;
        if (!current_counts.contains(class_id)) {
            disappearance_counter[class_id]++;
            if (disappearance_counter[class_id] >= disappearance_threshold) {
                it = previous_counts.erase(it);
                frame_stability.erase(class_id);
                disappearance_counter.erase(class_id);
                continue;
            }
        } else {
            disappearance_counter[class_id] = 0;
        }
        ++it;
    }

    for (const auto& [class_id, count] : current_counts) {
        previous_counts[class_id] = count;
    }
}

const std::map<int, int>& TotalCounter::getTotalCounts() {
    return total_counts;
}
