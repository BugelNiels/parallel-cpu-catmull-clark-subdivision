/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "mainview.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QGroupBox *SettingsGB;
    QPushButton *ImportOBJ;
    QGroupBox *generalGroupBox;
    QVBoxLayout *verticalLayout;
    QCheckBox *wireframeCheckBox;
    QCheckBox *requireApplyCheckBox;
    QCheckBox *showNormalsCheckBox;
    QLabel *StepsLabel;
    QSpinBox *SubdivSteps;
    QPushButton *applySubdivisionButton;
    QGroupBox *runtimeInfoBox;
    QLabel *timeElapsedLabel;
    QLabel *timeLabel;
    QPushButton *importDefaultButton;
    QGroupBox *groupBox;
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *h0Label;
    QLabel *h0LabelNum;
    QLabel *f0Label;
    QLabel *f0LabelNum;
    QLabel *e0Label;
    QLabel *e0LabelNum;
    QLabel *v0Label;
    QLabel *v0LabelNum;
    QSpacerItem *horizontalSpacer;
    QLabel *hdLabel;
    QLabel *fdLabel;
    QLabel *edLabel;
    QLabel *vdLabel;
    QLabel *hdLabelNum;
    QLabel *fdLabelNum;
    QLabel *edLabelNum;
    QLabel *vdLabelNum;
    MainView *MainDisplay;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1366, 768);
        MainWindow->setStyleSheet(QString::fromUtf8("/* Groupbox */\n"
"\n"
"QGroupBox {\n"
"    border: 1px solid #DDD;\n"
"    border-radius: 9px;\n"
"    margin-top: 9px;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 3px 0 3px;\n"
"}"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(6, 6, 6, 6);
        SettingsGB = new QGroupBox(centralWidget);
        SettingsGB->setObjectName(QString::fromUtf8("SettingsGB"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(SettingsGB->sizePolicy().hasHeightForWidth());
        SettingsGB->setSizePolicy(sizePolicy);
        SettingsGB->setMinimumSize(QSize(220, 0));
        ImportOBJ = new QPushButton(SettingsGB);
        ImportOBJ->setObjectName(QString::fromUtf8("ImportOBJ"));
        ImportOBJ->setGeometry(QRect(20, 30, 181, 28));
        generalGroupBox = new QGroupBox(SettingsGB);
        generalGroupBox->setObjectName(QString::fromUtf8("generalGroupBox"));
        generalGroupBox->setGeometry(QRect(20, 90, 181, 181));
        verticalLayout = new QVBoxLayout(generalGroupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        wireframeCheckBox = new QCheckBox(generalGroupBox);
        wireframeCheckBox->setObjectName(QString::fromUtf8("wireframeCheckBox"));

        verticalLayout->addWidget(wireframeCheckBox);

        requireApplyCheckBox = new QCheckBox(generalGroupBox);
        requireApplyCheckBox->setObjectName(QString::fromUtf8("requireApplyCheckBox"));

        verticalLayout->addWidget(requireApplyCheckBox);

        showNormalsCheckBox = new QCheckBox(generalGroupBox);
        showNormalsCheckBox->setObjectName(QString::fromUtf8("showNormalsCheckBox"));

        verticalLayout->addWidget(showNormalsCheckBox);

        StepsLabel = new QLabel(generalGroupBox);
        StepsLabel->setObjectName(QString::fromUtf8("StepsLabel"));

        verticalLayout->addWidget(StepsLabel);

        SubdivSteps = new QSpinBox(generalGroupBox);
        SubdivSteps->setObjectName(QString::fromUtf8("SubdivSteps"));

        verticalLayout->addWidget(SubdivSteps);

        applySubdivisionButton = new QPushButton(generalGroupBox);
        applySubdivisionButton->setObjectName(QString::fromUtf8("applySubdivisionButton"));

        verticalLayout->addWidget(applySubdivisionButton);

        runtimeInfoBox = new QGroupBox(SettingsGB);
        runtimeInfoBox->setObjectName(QString::fromUtf8("runtimeInfoBox"));
        runtimeInfoBox->setGeometry(QRect(20, 280, 181, 101));
        timeElapsedLabel = new QLabel(runtimeInfoBox);
        timeElapsedLabel->setObjectName(QString::fromUtf8("timeElapsedLabel"));
        timeElapsedLabel->setGeometry(QRect(10, 30, 111, 23));
        timeLabel = new QLabel(runtimeInfoBox);
        timeLabel->setObjectName(QString::fromUtf8("timeLabel"));
        timeLabel->setGeometry(QRect(10, 60, 71, 23));
        importDefaultButton = new QPushButton(SettingsGB);
        importDefaultButton->setObjectName(QString::fromUtf8("importDefaultButton"));
        importDefaultButton->setGeometry(QRect(20, 60, 181, 31));
        groupBox = new QGroupBox(SettingsGB);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(20, 390, 181, 291));
        formLayoutWidget = new QWidget(groupBox);
        formLayoutWidget->setObjectName(QString::fromUtf8("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 20, 160, 254));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setContentsMargins(0, 0, 0, 0);
        h0Label = new QLabel(formLayoutWidget);
        h0Label->setObjectName(QString::fromUtf8("h0Label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, h0Label);

        h0LabelNum = new QLabel(formLayoutWidget);
        h0LabelNum->setObjectName(QString::fromUtf8("h0LabelNum"));

        formLayout->setWidget(0, QFormLayout::FieldRole, h0LabelNum);

        f0Label = new QLabel(formLayoutWidget);
        f0Label->setObjectName(QString::fromUtf8("f0Label"));

        formLayout->setWidget(1, QFormLayout::LabelRole, f0Label);

        f0LabelNum = new QLabel(formLayoutWidget);
        f0LabelNum->setObjectName(QString::fromUtf8("f0LabelNum"));

        formLayout->setWidget(1, QFormLayout::FieldRole, f0LabelNum);

        e0Label = new QLabel(formLayoutWidget);
        e0Label->setObjectName(QString::fromUtf8("e0Label"));

        formLayout->setWidget(2, QFormLayout::LabelRole, e0Label);

        e0LabelNum = new QLabel(formLayoutWidget);
        e0LabelNum->setObjectName(QString::fromUtf8("e0LabelNum"));

        formLayout->setWidget(2, QFormLayout::FieldRole, e0LabelNum);

        v0Label = new QLabel(formLayoutWidget);
        v0Label->setObjectName(QString::fromUtf8("v0Label"));

        formLayout->setWidget(3, QFormLayout::LabelRole, v0Label);

        v0LabelNum = new QLabel(formLayoutWidget);
        v0LabelNum->setObjectName(QString::fromUtf8("v0LabelNum"));

        formLayout->setWidget(3, QFormLayout::FieldRole, v0LabelNum);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        formLayout->setItem(4, QFormLayout::LabelRole, horizontalSpacer);

        hdLabel = new QLabel(formLayoutWidget);
        hdLabel->setObjectName(QString::fromUtf8("hdLabel"));

        formLayout->setWidget(5, QFormLayout::LabelRole, hdLabel);

        fdLabel = new QLabel(formLayoutWidget);
        fdLabel->setObjectName(QString::fromUtf8("fdLabel"));

        formLayout->setWidget(6, QFormLayout::LabelRole, fdLabel);

        edLabel = new QLabel(formLayoutWidget);
        edLabel->setObjectName(QString::fromUtf8("edLabel"));

        formLayout->setWidget(7, QFormLayout::LabelRole, edLabel);

        vdLabel = new QLabel(formLayoutWidget);
        vdLabel->setObjectName(QString::fromUtf8("vdLabel"));

        formLayout->setWidget(8, QFormLayout::LabelRole, vdLabel);

        hdLabelNum = new QLabel(formLayoutWidget);
        hdLabelNum->setObjectName(QString::fromUtf8("hdLabelNum"));

        formLayout->setWidget(5, QFormLayout::FieldRole, hdLabelNum);

        fdLabelNum = new QLabel(formLayoutWidget);
        fdLabelNum->setObjectName(QString::fromUtf8("fdLabelNum"));

        formLayout->setWidget(6, QFormLayout::FieldRole, fdLabelNum);

        edLabelNum = new QLabel(formLayoutWidget);
        edLabelNum->setObjectName(QString::fromUtf8("edLabelNum"));

        formLayout->setWidget(7, QFormLayout::FieldRole, edLabelNum);

        vdLabelNum = new QLabel(formLayoutWidget);
        vdLabelNum->setObjectName(QString::fromUtf8("vdLabelNum"));

        formLayout->setWidget(8, QFormLayout::FieldRole, vdLabelNum);


        horizontalLayout->addWidget(SettingsGB);

        MainDisplay = new MainView(centralWidget);
        MainDisplay->setObjectName(QString::fromUtf8("MainDisplay"));
        MainDisplay->setMouseTracking(true);

        horizontalLayout->addWidget(MainDisplay);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        SettingsGB->setTitle(QApplication::translate("MainWindow", "Settings", nullptr));
        ImportOBJ->setText(QApplication::translate("MainWindow", "Import OBJ file", nullptr));
        wireframeCheckBox->setText(QApplication::translate("MainWindow", "Wireframe", nullptr));
        requireApplyCheckBox->setText(QApplication::translate("MainWindow", "Require Apply", nullptr));
        showNormalsCheckBox->setText(QApplication::translate("MainWindow", "Show Normals", nullptr));
        StepsLabel->setText(QApplication::translate("MainWindow", "Catmull-Clark steps", nullptr));
        applySubdivisionButton->setText(QApplication::translate("MainWindow", "Apply", nullptr));
        runtimeInfoBox->setTitle(QApplication::translate("MainWindow", "Runtime Info", nullptr));
        timeElapsedLabel->setText(QApplication::translate("MainWindow", "Time Elapsed:", nullptr));
        timeLabel->setText(QString());
        importDefaultButton->setText(QApplication::translate("MainWindow", "Import Default", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "Mesh Info", nullptr));
        h0Label->setText(QApplication::translate("MainWindow", "H0", nullptr));
        h0LabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        f0Label->setText(QApplication::translate("MainWindow", "F0", nullptr));
        f0LabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        e0Label->setText(QApplication::translate("MainWindow", "E0", nullptr));
        e0LabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        v0Label->setText(QApplication::translate("MainWindow", "V0", nullptr));
        v0LabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        hdLabel->setText(QApplication::translate("MainWindow", "HD", nullptr));
        fdLabel->setText(QApplication::translate("MainWindow", "FD", nullptr));
        edLabel->setText(QApplication::translate("MainWindow", "ED", nullptr));
        vdLabel->setText(QApplication::translate("MainWindow", "VD", nullptr));
        hdLabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        fdLabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        edLabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
        vdLabelNum->setText(QApplication::translate("MainWindow", "0", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
