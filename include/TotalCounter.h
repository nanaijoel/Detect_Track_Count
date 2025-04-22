#pragma once

#include <map>
#include <set>
#include <mutex>
#include <memory>
#include <vector>
#include <deque>


namespace byte_track {
    class BYTETracker;
    class STrack;
}


extern std::map<int, int> total_counts;


struct TrackHistory {
    std::deque<int> recent_classes;
    bool was_crossing = false;
};

class TotalCounter {
public:
    void update(const std::vector<std::shared_ptr<byte_track::STrack>>& tracks, int scanLineX,
        const std::vector<int>& filter_classes);

    std::map<int, int> getCounts() const;

private:
    std::set<int> active_ids;
    std::map<int, bool> was_crossing;
    mutable std::mutex mutex;
    std::map<int, TrackHistory> history_map;

};
