#ifndef OBJECT_DETECTION_GUI_H
#define OBJECT_DETECTION_GUI_H

#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QImage>
#include <QTimer>
#include <QResizeEvent>
#include <opencv2/opencv.hpp>
#include "Detect_and_Draw.h"

// Neue globale Variable zur Klassenauswahl
extern std::vector<int> active_classes;

class ObjectDetectionGUI : public QWidget {
    Q_OBJECT

public:
    explicit ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent = nullptr);
    ~ObjectDetectionGUI() override = default;
    void updateFrame(const cv::Mat& frame) const;

    private slots:
        void handleReset() const;
    void handleClassSelection(int index);

private:
    void updateCounts() const;
    static QImage matToQImage(const cv::Mat& mat);

    DetectAndDraw* detector;

    QLabel* frameLabel;
    QLabel* actualCountLabels[3];
    QLabel* totalCountLabels[3];
    QPushButton* resetButton;
    QComboBox* classSelector;
};

#endif // OBJECT_DETECTION_GUI_H
