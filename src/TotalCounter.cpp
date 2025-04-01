#include "TotalCounter.h"


std::map<int, int> total_counts = {{0, 0}, {1, 0}, {2, 0}};

void TotalCounter::update(const std::vector<int>& current_classIds) {
    // 1. Zähle aktuelle Klassen im Frame
    std::map<int, int> current_counts;
    for (int id : current_classIds) {
        current_counts[id]++;
    }

    // 2. Für jede Klasse prüfen, ob mehr Objekte hinzugekommen sind
    for (const auto& [class_id, current_count] : current_counts) {
        int previous_count = previous_counts[class_id];

        if (current_count > previous_count) {
            // Es sind neue Objekte hinzugekommen → Stabilitätswert erhöhen
            frame_stability[class_id]++;
            if (frame_stability[class_id] >= stability_threshold) {
                int delta = current_count - previous_count;
                total_counts[class_id] += delta;
                frame_stability[class_id] = 0; // Nach dem Zählen zurücksetzen
            }
        } else {
            // Keine Zunahme → Stabilität resetten
            frame_stability[class_id] = 0;
        }
    }

    // 3. Update der vorherigen Zählung → aber NUR für Klassen, die auch aktuell da sind
    previous_counts = current_counts;
}

const std::map<int, int>& TotalCounter::getTotalCounts() const {
    return total_counts;
}
