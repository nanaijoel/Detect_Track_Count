#pragma once
#include <map>
#include <set>
#include <unordered_map>
#include "BYTETracker.h"
#include <unordered_set>

extern std::map<int, int> total_counts;

class TotalCounter {
public:
    void update(const std::vector<std::shared_ptr<class byte_track::STrack>>& tracks);

private:
    std::unordered_map<int, int> id_to_class;
    std::unordered_map<int, int> id_stability;
    std::set<int> already_counted_ids;
    std::unordered_set<size_t> counted_ids;
    int min_frames_to_count = 5;

    const int stability_threshold = 5; // z. B. 5 Frames
};
