#pragma once
#include <map>
#include <vector>

extern std::map<int, int> total_counts;

class TotalCounter {
public:
    void update(const std::vector<int>& current_classIds);
    [[nodiscard]] const std::map<int, int>& getTotalCounts() const;

private:
    std::map<int, int> frame_stability;
    std::map<int, int> previous_counts;
    const int stability_threshold = 1;
};
