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
    std::unordered_set<size_t> counted_ids;
    int min_frames_to_count = 3;

};
