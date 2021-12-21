#-------------------------------------------------
#
# Project created by QtCreator 2016-12-08T16:31:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CatmarkSubdiv
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
  meshinitializer.cpp \
    meshrenderer.cpp \
    objfile.cpp \
    mesh.cpp \
    meshtools.cpp \
    mainview.cpp \
    settings.cpp

HEADERS  += mainwindow.h \
    mesh.h \
    meshinitializer.h \
    meshrenderer.h \
    objfile.h \
    renderer.h \
    settings.h \
    mainview.h

FORMS    += mainwindow.ui

RESOURCES += \
    resources.qrc
