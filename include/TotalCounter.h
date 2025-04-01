#pragma once
#include <map>
#include <vector>

extern std::map<int, int> total_counts;

class TotalCounter {
public:
    void update(const std::vector<int>& current_classIds);
    [[nodiscard]] static const std::map<int, int>& getTotalCounts();

private:
    std::map<int, int> frame_stability;
    std::map<int, int> previous_counts;
    std::map<int, int> disappearance_counter;

    const int stability_threshold = 1;
    const int disappearance_threshold = 15;
};
