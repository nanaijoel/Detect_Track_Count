#pragma once

#include <map>
#include <set>
#include <mutex>
#include <memory>
#include <vector>

namespace byte_track {
    class BYTETracker;
    class STrack;
}


extern std::map<int, int> total_counts;

class TotalCounter {
public:
    void update(const std::vector<std::shared_ptr<byte_track::STrack>>& tracks, int scanLineX);

    std::map<int, int> getCounts() const;

private:
    std::set<int> active_ids;  // IDs, die aktuell über die Linie sind
    std::map<int, bool> was_crossing;
    mutable std::mutex mutex;
};
