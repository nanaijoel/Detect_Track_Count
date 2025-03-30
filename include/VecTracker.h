#ifndef VEC_TRACKER_H
#define VEC_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>

class VecTracker {
private:
    cv::Rect bounding_box;
    int id = -1;
    int class_id = -1;
    int age = 0;

    std::vector<float> relative_distances;

    int missing_counter = 0;  // ⬅️ NEU: zählt, wie lange das Objekt fehlt
    bool active = true;       // ⬅️ NEU: ob der Tracker aktuell benutzt wird
    static int next_id;


public:
    VecTracker(const cv::Rect& bbox);

    cv::Point get_center() const;
    cv::Rect get_bounding_box() const;
    int get_age() const;
    void increment_age();
    void reset_age();

    void compute_relative_distances(const std::vector<cv::Rect>& all_bboxes);

    const std::vector<float>& get_relative_distances() const;

    int get_id() const;
    void set_id(int new_id);

    int get_class_id() const;
    void set_class_id(int cid);

    // ⬇️ NEU:
    void increment_missing() { ++missing_counter; }
    void reset_missing() { missing_counter = 0; }
    int get_missing_count() const { return missing_counter; }

    void deactivate() { active = false; }
    bool is_active() const { return active; }

    bool is_potential_match(const VecTracker& other, float tolerance) const;

    static std::vector<VecTracker> update_trackers(
        const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classIds,
        std::vector<VecTracker>& previous_trackers
    );

};


#endif // VEC_TRACKER_H
