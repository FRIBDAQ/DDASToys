TEMPLATE = app
  
QT += widgets

# CXXFLAGS from $ROOTSYS/bin/root-config --cflags
QMAKE_CXXFLAGS += -pthread -std=c++14 -m64
CONFIG += qt warn_on thread console

INCLUDEPATH += .. $(DAQINC) $(DDAS_INC) $(ROOTSYS)/include

# NSCLDAQ and DDAS libraries
LIBS += -L$(DAQLIB) -lFragmentIndex -ldataformat -ldaqio -lException -lurl -Wl,-rpath=$(DAQLIB) -L$(DDAS_LIB) -lddasformat -lddasfitformat -Wl,-rpath=$(DDAS_LIB)

# ROOT libraries from $ROOTSYS/bin/root-config --libs --ldflags 
LIBS += -L$(ROOTSYS)/lib -lCore -lRIO -lNet \
        -lHist -lGraf -lGraf3d -lGpad -lTree \
        -lRint -lPostscript -lMatrix -lPhysics \
        -lGui -lRGL

# Object files to link from the top build directory
LIBS += ../DDASFitHitUnpacker.o ../Configuration.o ../functions_analytic.o ../functions_template.o

HEADERS += QTraceView.h QHitData.h QRootCanvas.h DDASRingItemProcessor.h DDASDecoder.h FitManager.h
SOURCES += main.cpp QTraceView.cpp QHitData.cpp QRootCanvas.cpp DDASRingItemProcessor.cpp DDASDecoder.cpp FitManager.cpp
