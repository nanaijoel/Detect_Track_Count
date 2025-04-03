#include "TotalCounter.h"
#include "STrack.h"

#include <iostream>


std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};


void TotalCounter::update(const std::vector<std::shared_ptr<byte_track::STrack>>& tracks, int scanLineX)
{
std::lock_guard<std::mutex> lock(mutex);

for (const auto& track : tracks){
    if (!track->isActivated()) continue;
    if (track->getSTrackState() != byte_track::STrackState::Tracked) continue;

    const auto& box = track->getRect();
    int track_id = static_cast<int>(track->getTrackId());
    int class_id = track->getClassId();

    int left_x = static_cast<int>(box.tl_x());
    int right_x = static_cast<int>(box.br_x());


    bool is_crossing = (left_x <= scanLineX && right_x >= scanLineX);
    bool was = was_crossing[track_id];

    if (is_crossing && !was){
    total_counts[class_id]++;
    }

    was_crossing[track_id] = is_crossing;
    }
}
