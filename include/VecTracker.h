#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

class VecTracker {
public:
    VecTracker(const cv::Rect& box, int class_id);

    void update_box(const cv::Rect& new_box);
    void compute_direction_vectors(const std::vector<VecTracker>& all_objects);
    [[nodiscard]] bool is_similar(const VecTracker& other, float tolerance) const;

    [[nodiscard]] const cv::Rect& get_bounding_box() const { return current_box; }
    [[nodiscard]] int get_class_id() const { return class_id; }
    void set_class_id(int id) { class_id = id; }

    static std::vector<VecTracker> update_trackers(
        const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classIds,
        const std::vector<VecTracker>& previous_trackers,
        float tolerance = 2.0f
    );

private:
    cv::Rect last_box;
    cv::Rect current_box;
    int class_id;
    std::vector<cv::Point> direction_vectors;
};
