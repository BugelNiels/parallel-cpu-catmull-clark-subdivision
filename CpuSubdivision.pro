#-------------------------------------------------
#
# Project created by QtCreator 2016-12-08T16:31:55
#
#-------------------------------------------------

QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = cpuSubdivide
TEMPLATE = app

CONFIG += release

release: DESTDIR = build/
debug:   DESTDIR = build/

INCLUDEPATH += src/

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.qrc
UI_DIR = $$DESTDIR/.ui

SOURCES += src/main.cpp\
    src/mainwindow.cpp \
    src/meshSubdivision.cpp \
    src/meshinitializer.cpp \
    src/meshrenderer.cpp \
    src/objfile.cpp \
    src/mesh.cpp \
    src/mainview.cpp \
    src/quadmesh.cpp \
    src/settings.cpp \
    src/subdivider.cpp \
    src/util.cpp

HEADERS  += src/mainwindow.h \
    src/mesh.h \
    src/meshinitializer.h \
    src/meshrenderer.h \
    src/objfile.h \
    src/quadmesh.h \
    src/renderer.h \
    src/settings.h \
    src/mainview.h \
    src/subdivider.h \
    src/util.h

FORMS += src/mainwindow.ui

RESOURCES += \
    src/resources.qrc


QMAKE_CXXFLAGS+= -fopenmp -lpthread
QMAKE_LFLAGS +=  -fopenmp -lpthread

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
