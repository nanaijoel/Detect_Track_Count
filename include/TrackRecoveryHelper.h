#ifndef TRACK_RECOVERY_HELPER_H
#define TRACK_RECOVERY_HELPER_H

#include <opencv2/core.hpp>
#include <vector>
#include "Sort.h"

class TrackRecoveryHelper {
public:
    struct RecoveryCandidate {
        int track_id;
        cv::Rect predicted_box;
        int class_id;
        int missing_frames;

        RecoveryCandidate(int id, const cv::Rect& box, int cls)
            : track_id(id), predicted_box(box), class_id(cls), missing_frames(0) {}
    };

    void store_unmatched_track(const SORT::Track& track);
    void update_missing();
    void try_recover(const std::vector<cv::Rect>& detections,
                     const std::vector<int>& classIds,
                     std::vector<bool>& matched,
                     std::vector<SORT::Track>& tracks);

    bool try_recover_single(const cv::Rect& new_box,
                            int new_class,
                            float confidence,
                            std::vector<SORT::Track>& tracks,
                            int& reuse_id);

private:
    std::vector<RecoveryCandidate> buffer;
    const int max_missing_frames = 8;
    const float max_distance = 35.0f;
};

#endif // TRACK_RECOVERY_HELPER_H