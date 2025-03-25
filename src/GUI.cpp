#include "GUI.h"
#include <QHBoxLayout>


ObjectDetectionGUI::ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent)
: QWidget(parent), detector(detector), actualCountLabels{nullptr}, totalCountLabels{nullptr}  {

    frameLabel = new QLabel(this);
    frameLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    frameLabel->setAlignment(Qt::AlignCenter);

    auto* rightLayout = new QVBoxLayout;


    auto* titleLabel = new QLabel("OBJECT COUNTS", this);
    titleLabel->setStyleSheet("font-size: 20pt; font-weight: bold; color: blue;");
    titleLabel->setAlignment(Qt::AlignCenter);
    rightLayout->addWidget(titleLabel);
    rightLayout->addSpacing(15);


    QString classes[3] = {"Bear", "Frog", "Cola"};
    for (int i = 0; i < 3; ++i) {
        actualCountLabels[i] = new QLabel(QString("%1 Actual: 0").arg(classes[i]));
        actualCountLabels[i]->setStyleSheet("font-size: 16pt; color: cyan;");
        actualCountLabels[i]->setAlignment(Qt::AlignCenter);
        rightLayout->addWidget(actualCountLabels[i]);
    }

    rightLayout->addSpacing(30);


    for (int i = 0; i < 3; ++i) {
        totalCountLabels[i] = new QLabel(QString("%1 Total: 0").arg(classes[i]));
        totalCountLabels[i]->setStyleSheet("font-size: 16pt; color: lightblue;");
        totalCountLabels[i]->setAlignment(Qt::AlignCenter);
        rightLayout->addWidget(totalCountLabels[i]);
    }


    rightLayout->addSpacing(30);

    resetButton = new QPushButton("Reset Total Counts", this);
    resetButton->setStyleSheet("font-size: 12pt; padding: 6px; border: 2px solid white; color: white;");
    resetButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(resetButton, SIGNAL(clicked()), this, SLOT(handleReset()));
    rightLayout->addWidget(resetButton);
    rightLayout->setAlignment(resetButton, Qt::AlignCenter);
    rightLayout->addStretch();

    auto* mainLayout = new QHBoxLayout(this);
    mainLayout->addWidget(frameLabel, 3);
    mainLayout->addLayout(rightLayout, 1);
    setLayout(mainLayout);
    setStyleSheet("background-color: black; color: white;");
}

void ObjectDetectionGUI::updateFrame(const cv::Mat& frame) const {
    if (!frame.empty()) {
        QImage image = matToQImage(frame);
        frameLabel->setPixmap(QPixmap::fromImage(image).scaled(frameLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        updateCounts();
    }
}

void ObjectDetectionGUI::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    QPixmap pix = frameLabel->pixmap();
    if (!pix.isNull()) {
        frameLabel->setPixmap(pix.scaled(frameLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void ObjectDetectionGUI::handleReset() const {
    if (detector) {
        DetectAndDraw::reset_counts();
    }
}

void ObjectDetectionGUI::updateCounts() const {
    std::lock_guard<std::mutex> lock(count_mutex);
    QString classes[3] = {"Bear", "Frog", "Cola"};
    for (int i = 0; i < 3; ++i) {
        actualCountLabels[i]->setText(QString("%1 Actual: %2").arg(classes[i]).arg(actual_counts[i]));
        totalCountLabels[i]->setText(QString("%1 Total: %2").arg(classes[i]).arg(total_counts[i]));
    }
}

QImage ObjectDetectionGUI::matToQImage(const cv::Mat& mat) {
    if (mat.channels() == 3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
    } else if (mat.channels() == 1) {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    } else {
        return {};
    }
}
