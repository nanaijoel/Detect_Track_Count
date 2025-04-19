#include <ranges>
#include "TotalCounter.h"
#include "BYTETracker/STrack.h"

std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};

constexpr int HISTORY_LENGTH = 5;
constexpr int CLASS_STABILITY_THRESHOLD = 3;

void TotalCounter::update(const std::vector<std::shared_ptr<byte_track::STrack>>& tracks, int scanLineX, const std::vector<int>& filter_classes)
{
    std::lock_guard<std::mutex> lock(mutex);

    for (const auto& track : tracks)
    {
        if (!track->isActivated()) continue;
        if (track->getSTrackState() != byte_track::STrackState::Tracked) continue;

        const auto& box = track->getRect();
        int track_id = static_cast<int>(track->getTrackId());
        int class_id = track->getClassId();

        if (!filter_classes.empty() &&
            std::ranges::find(filter_classes, class_id) == filter_classes.end()) {
            continue;
            }


        int left_x = static_cast<int>(box.tl_x());
        int right_x = static_cast<int>(box.br_x());

        bool is_crossing = (left_x <= scanLineX && right_x >= scanLineX);
        bool was = history_map[track_id].was_crossing;

        auto& history = history_map[track_id].recent_classes;
        history.push_back(class_id);
        if (history.size() > HISTORY_LENGTH)
            history.pop_front();

        if (is_crossing && !was)
        {
            std::map<int, int> class_counter;
            for (int cid : history) class_counter[cid]++;
            int most_common_class = -1;
            int max_count = 0;
            for (const auto& [cid, count] : class_counter)
            {
                if (count > max_count)
                {
                    max_count = count;
                    most_common_class = cid;
                }
            }

            if (max_count >= CLASS_STABILITY_THRESHOLD)
            {
                total_counts[most_common_class]++;
            }
        }

        history_map[track_id].was_crossing = is_crossing;
    }

    std::set<int> current_ids;
    for (const auto& track : tracks) {
        if (track->isActivated() && track->getSTrackState() == byte_track::STrackState::Tracked) {
            current_ids.insert(static_cast<int>(track->getTrackId()));
        }
    }

    for (auto it = history_map.begin(); it != history_map.end(); ) {
        if (!current_ids.contains(it->first)) {
            it = history_map.erase(it);
        } else {
            ++it;
        }
    }
}

std::map<int, int> TotalCounter::getCounts() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return total_counts;
}