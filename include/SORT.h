#ifndef SORT_H
#define SORT_H

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include <map>
#include <vector>
#include <set>
#include <atomic>

// **Globale Variablen für Tracking & Multithreading**
extern std::mutex count_mutex;
extern std::atomic<bool> stopThreads;
extern std::map<int, int> total_counts;  // Gesamtanzahl aller erkannten Objekte
extern std::map<int, int> actual_counts; // Anzahl der aktuell erkannten Objekte

// **SORT-Tracker-Klasse**
class SORT {
public:
    struct Track {
        cv::KalmanFilter kf;
        cv::Rect box;
        int id;
        int classId;
        int frames_since_seen;
        bool matched_in_this_frame;

        Track(int track_id, cv::Rect bbox, int class_id);
        void update(cv::Rect new_box);
        void predict();
    };

    std::vector<Track> tracks;
    int next_id = 0;

    void update_tracks(const std::vector<cv::Rect>& detected_boxes, const std::vector<int>& classIds);
    std::vector<Track> get_tracks();
};

#endif // SORT_H
