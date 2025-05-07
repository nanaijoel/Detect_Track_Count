#include <QHBoxLayout>
#include <QGuiApplication>
#include <QStandardItem>
#include <QScreen>
#include "GUI.h"
#include "CameraMode.h"
#include "TotalCounter.h"



std::vector<int> active_classes = {0, 1, 2};  // default: all

ObjectDetectionGUI::ObjectDetectionGUI(DetectAndDraw* detector, QWidget* parent)
    : QWidget(parent),
      detector(detector),
      frameLabel(nullptr),
      actualCountLabels{nullptr},
      totalCountLabels{nullptr},
      resetButton(nullptr),
      classSelector(nullptr)
{
    setupLayout();
}


void ObjectDetectionGUI::setupLayout() {
    int screenHeight = QGuiApplication::primaryScreen()->size().height();
    int titleFontSize = screenHeight / 40;
    int labelFontSize = screenHeight / 55;
    int buttonFontSize = screenHeight / 65;

    frameLabel = new QLabel(this);
    frameLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    frameLabel->setAlignment(Qt::AlignCenter);

    auto* rightLayout = new QVBoxLayout;
    setupTitle(rightLayout, titleFontSize);
    setupCountLabels(rightLayout, labelFontSize);
    setupClassSelector(rightLayout, buttonFontSize);
    setupResetButton(rightLayout, buttonFontSize);
    rightLayout->addStretch();

    auto* mainLayout = new QHBoxLayout(this);
    mainLayout->addWidget(frameLabel, 3);
    mainLayout->addLayout(rightLayout, 1);
    setLayout(mainLayout);

    setStyleSheet("background-color: black; color: white;");
}

void ObjectDetectionGUI::setupTitle(QVBoxLayout* layout, int fontSize) {
    auto* titleLabel = new QLabel("OBJECT COUNTS", this);
    titleLabel->setStyleSheet(QString("font-size: %1pt; font-weight: bold; color: blue;").arg(fontSize));
    titleLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(titleLabel);
    layout->addSpacing(15);
}

void ObjectDetectionGUI::setupCountLabels(QVBoxLayout* layout, int fontSize) {
    QString classes[3] = {"Bear", "Frog", "Cola"};
    for (int i = 0; i < 3; ++i) {
        actualCountLabels[i] = new QLabel(QString("%1 Actual: 0").arg(classes[i]), this);
        actualCountLabels[i]->setStyleSheet(QString("font-size: %1pt; color: cyan;").arg(fontSize));
        actualCountLabels[i]->setAlignment(Qt::AlignCenter);
        layout->addWidget(actualCountLabels[i]);
    }

    layout->addSpacing(30);

    for (int i = 0; i < 3; ++i) {
        totalCountLabels[i] = new QLabel(QString("%1 Total: 0").arg(classes[i]), this);
        totalCountLabels[i]->setStyleSheet(QString("font-size: %1pt; color: lightblue;").arg(fontSize));
        totalCountLabels[i]->setAlignment(Qt::AlignCenter);
        layout->addWidget(totalCountLabels[i]);
    }

    layout->addSpacing(30);
}

void ObjectDetectionGUI::setupClassSelector(QVBoxLayout* layout, int fontSize) {
    classSelector = new QComboBox(this);
    classSelector->addItems({"Detect All", "Only Bears", "Only Frogs", "Only Colas"});
    classSelector->setStyleSheet(QString("QComboBox { font-size: %1pt; color: white; background-color: black; }").arg(fontSize));

    auto model = qobject_cast<QStandardItemModel*>(classSelector->model());
    for (int i = 0; i < classSelector->count(); ++i) {
        if (QStandardItem* item = model->item(i)) {
            item->setTextAlignment(Qt::AlignCenter);
        }
    }

    connect(classSelector, SIGNAL(currentIndexChanged(int)), this, SLOT(handleClassSelection(int)));
    layout->addWidget(classSelector);
    layout->setAlignment(classSelector, Qt::AlignCenter);
    layout->addSpacing(30);
}

void ObjectDetectionGUI::setupResetButton(QVBoxLayout* layout, int fontSize) {
    resetButton = new QPushButton("Reset Total Counts", this);
    resetButton->setStyleSheet(QString("font-size: %1pt; padding: 6px; border: 2px solid white; color: white;").arg(fontSize));
    resetButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    connect(resetButton, SIGNAL(clicked()), this, SLOT(handleReset()));
    layout->addWidget(resetButton);
    layout->setAlignment(resetButton, Qt::AlignCenter);
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

void ObjectDetectionGUI::handleReset() const {
    if (detector) {
        DetectAndDraw::reset_counts();
    }
}

void ObjectDetectionGUI::handleClassSelection(int index) {
    switch (index) {
        case 0: active_classes = {0, 1, 2}; break;
        case 1: active_classes = {0}; break;
        case 2: active_classes = {1}; break;
        case 3: active_classes = {2}; break;
        default: ;
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
