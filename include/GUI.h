#ifndef OBJECT_DETECTION_GUI_H
#define OBJECT_DETECTION_GUI_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QImage>
#include <QTimer>
#include <QResizeEvent>
#include <opencv2/opencv.hpp>
#include "Detect_and_Draw.h"

class ObjectDetectionGUI : public QWidget {
    Q_OBJECT

public:
    explicit ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent = nullptr);
    ~ObjectDetectionGUI() override = default;  // ✅ virtueller Destruktor

    void updateFrame(const cv::Mat& frame);

protected:
    void resizeEvent(QResizeEvent* event) override;

    private slots:
        void handleReset() const;

private:
    void updateCounts() const;
    QImage matToQImage(const cv::Mat& mat);

    DetectAndDraw* detector;

    QLabel* frameLabel;
    QLabel* actualCountLabels[3];
    QLabel* totalCountLabels[3];
    QPushButton* resetButton;
};

#endif // OBJECT_DETECTION_GUI_H
