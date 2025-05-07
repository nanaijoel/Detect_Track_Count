#ifndef OBJECT_DETECTION_GUI_H
#define OBJECT_DETECTION_GUI_H

#include <QWidget>
#include <QHBoxLayout>
#include <QStandardItemModel>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QImage>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>
#include "Detect_and_Draw.h"


extern std::vector<int> active_classes;

class ObjectDetectionGUI : public QWidget {
    Q_OBJECT

public:
    explicit ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent = nullptr);
    ~ObjectDetectionGUI() override = default;

    void updateFrame(const cv::Mat& frame) const;

    private slots:
        void handleReset() const;

    static void handleClassSelection(int index);

private:
    void setupLayout();
    void setupTitle(QVBoxLayout* layout, int fontSize);
    void setupCountLabels(QVBoxLayout* layout, int fontSize);
    void setupClassSelector(QVBoxLayout* layout, int fontSize);
    void setupResetButton(QVBoxLayout* layout, int fontSize);

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
