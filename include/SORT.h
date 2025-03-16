#ifndef SORT_H
#define SORT_H

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>
#include <atomic>

extern std::mutex count_mutex;
extern std::atomic<bool> stopThreads;
extern std::map<int, int> total_counts;
extern std::map<int, int> actual_counts;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
    }
};

class SORT {
public:
    struct Track {
        cv::KalmanFilter kf;
        cv::Rect box;
        int id;
        int classId;
        int frames_since_seen;
        int frames_since_creation; // Neu: Erst nach mehreren Frames zählen
        bool matched_in_this_frame;
        bool was_counted; // Neu: Um doppelte Zählung zu verhindern

        Track(int track_id, cv::Rect bbox, int class_id);
        void update(cv::Rect new_box);
        void predict();
    };

    std::vector<Track> tracks;
    int next_id = 0;

    void update_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds);
    [[nodiscard]] std::vector<Track> get_tracks() const;

private:
    std::unordered_map<std::pair<int, int>, float, pair_hash> previous_distances;

    void update_distances();
    // bool validate_object_identity(const Track& new_track);
    void match_existing_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds, std::vector<bool>& matched);
    void add_new_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds, std::vector<bool>& matched);
    void remove_old_tracks();
    void update_counts();

};

#endif // SORT_H
