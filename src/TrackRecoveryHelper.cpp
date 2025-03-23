#include "TrackRecoveryHelper.h"
#include <cmath>
#include <algorithm>

void TrackRecoveryHelper::store_unmatched_track(const SORT::Track& track) {
    buffer.emplace_back(track.id, track.box, track.classId);
}

void TrackRecoveryHelper::update_missing() {
    for (auto& candidate : buffer) {
        candidate.missing_frames++;
    }

    std::erase_if(buffer,
                  [this](const RecoveryCandidate& c) {
                      return c.missing_frames > max_missing_frames;
                  });
}

void TrackRecoveryHelper::try_recover(const std::vector<cv::Rect>& detections,
                                      const std::vector<int>& classIds,
                                      std::vector<bool>& matched,
                                      std::vector<SORT::Track>& tracks) {
    for (auto it = buffer.begin(); it != buffer.end();) {
        bool recovered = false;

        for (size_t i = 0; i < detections.size(); ++i) {
            if (matched[i]) continue;

            float det_center_x = static_cast<float>(detections[i].x) + static_cast<float>(detections[i].width) / 2.0f;
            float det_center_y = static_cast<float>(detections[i].y) + static_cast<float>(detections[i].height) / 2.0f;

            float cand_center_x = static_cast<float>(it->predicted_box.x) + static_cast<float>(it->predicted_box.width) / 2.0f;
            float cand_center_y = static_cast<float>(it->predicted_box.y) + static_cast<float>(it->predicted_box.height) / 2.0f;

            float dx = det_center_x - cand_center_x;
            float dy = det_center_y - cand_center_y;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= max_distance && classIds[i] == it->class_id) {
                SORT::Track recovered_track(it->track_id, detections[i], classIds[i]);
                tracks.push_back(recovered_track);

                matched[i] = true;
                recovered = true;
                break;
            }
        }

        if (recovered) {
            it = buffer.erase(it);
        } else {
            ++it;
        }
    }
}

bool TrackRecoveryHelper::try_recover_single(const cv::Rect& new_box,
                                             int new_class,
                                             float /*confidence*/,
                                             std::vector<SORT::Track>& tracks,
                                             int& reuse_id) {
    cv::Point2f new_center(
        static_cast<float>(new_box.x + new_box.width) / 2.0f,
        static_cast<float>(new_box.y + new_box.height) / 2.0f
    );

    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
        float cand_center_x = static_cast<float>(it->predicted_box.x) + static_cast<float>(it->predicted_box.width) / 2.0f;
        float cand_center_y = static_cast<float>(it->predicted_box.y) + static_cast<float>(it->predicted_box.height) / 2.0f;

        float dx = new_center.x - cand_center_x;
        float dy = new_center.y - cand_center_y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist <= max_distance && new_class == it->class_id) {
            reuse_id = it->track_id;
            buffer.erase(it);
            return true;
        }
    }

    return false;
}
