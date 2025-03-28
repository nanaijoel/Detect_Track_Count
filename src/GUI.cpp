#include "GUI.h"
#include <QHBoxLayout>
#include <QScreen>

ObjectDetectionGUI::ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent)
    : QWidget(parent), detector(detector), actualCountLabels{nullptr}, totalCountLabels{nullptr} {

    int screenHeight = QGuiApplication::primaryScreen()->size().height();
    int titleFontSize = screenHeight / 40;
    int labelFontSize = screenHeight / 55;
    int buttonFontSize = screenHeight / 65;

    frameLabel = new QLabel(this);
    frameLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    frameLabel->setAlignment(Qt::AlignCenter);

    auto* rightLayout = new QVBoxLayout;

    auto* titleLabel = new QLabel("OBJECT COUNTS", this);
    titleLabel->setStyleSheet(QString("font-size: %1pt; font-weight: bold; color: blue;").arg(titleFontSize));
    titleLabel->setAlignment(Qt::AlignCenter);
    rightLayout->addWidget(titleLabel);
    rightLayout->addSpacing(15);

    QString classes[3] = {"Bear", "Frog", "Cola"};
    for (int i = 0; i < 3; ++i) {
        actualCountLabels[i] = new QLabel(QString("%1 Actual: 0").arg(classes[i]), this);
        actualCountLabels[i]->setStyleSheet(QString("font-size: %1pt; color: cyan;").arg(labelFontSize));
        actualCountLabels[i]->setAlignment(Qt::AlignCenter);
        rightLayout->addWidget(actualCountLabels[i]);
    }

    rightLayout->addSpacing(30);

    for (int i = 0; i < 3; ++i) {
        totalCountLabels[i] = new QLabel(QString("%1 Total: 0").arg(classes[i]), this);
        totalCountLabels[i]->setStyleSheet(QString("font-size: %1pt; color: lightblue;").arg(labelFontSize));
        totalCountLabels[i]->setAlignment(Qt::AlignCenter);
        rightLayout->addWidget(totalCountLabels[i]);
    }

    rightLayout->addSpacing(30);

    resetButton = new QPushButton("Reset Total Counts", this);
    resetButton->setStyleSheet(QString("font-size: %1pt; padding: 6px; border: 2px solid white; color: white;").arg(buttonFontSize));
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
        const QImage image = matToQImage(frame);
        frameLabel->setPixmap(QPixmap::fromImage(image).scaled(
            frameLabel->size(),
            Qt::KeepAspectRatio,
            Qt::FastTransformation
        ));
        updateCounts();
    }
}

// void ObjectDetectionGUI::resizeEvent(QResizeEvent* event) {
//     QWidget::resizeEvent(event);
//     const QPixmap pix = frameLabel->pixmap();
//     if (!pix.isNull()) {
//         frameLabel->setPixmap(pix.scaled(
//             frameLabel->size(),
//             Qt::KeepAspectRatio,
//             Qt::SmoothTransformation
//         ));
//     }
// }

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
    }
    return {};
}
