#include "TotalCounter.h"
#include "STrack.h"
#include <set>
#include <iostream>

std::set<size_t> counted_ids;
std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};

void TotalCounter::update(const std::vector<byte_track::BYTETracker::STrackPtr>& tracks)
{
    for (const auto& track : tracks) {
        if (!track->isActivated()) continue;
        if (track->getSTrackState() != byte_track::STrackState::Tracked) continue;
        if (track->getTrackletLength() < min_frames_to_count) continue;

        int id = static_cast<int>(track->getTrackId());
        int class_id = track->getClassId();

        if (counted_ids.find(id) == counted_ids.end()) {
            total_counts[class_id]++;
            counted_ids.insert(id);
            std::cout << "[ZÄHLUNG] Class " << class_id << ", ID " << id << " → gezählt!\n";
        }
    }
}

