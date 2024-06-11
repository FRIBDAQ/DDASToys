TEMPLATE = app
  
QT += widgets

QMAKE_CXXFLAGS += $(shell $(ROOTSYS)/bin/root-config --cflags)
CONFIG += qt warn_on thread console

isEmpty(FMTINC) {
FMTINC=../DDASFormat
}
isEmpty(FMTLIB) {
FMTLIB=../DDASFormat/build
}
message(Using FMTINC="$$FMTINC")
message(Using FMTLIB="$$FMTLIB")

INCLUDEPATH += .. $${FMTINC} $(DAQINC) $(ROOTSYS)/include

## FRIBDAQ libraries

LIBS += -L$(DAQLIB) -lFragmentIndex -ldataformat -ldaqio -lException -lurl -Wl,-rpath=$(DAQLIB) -L$${FMTLIB} -lDDASFormat -Wl,-rpath=$${FMTLIB}

## ROOT libraries

LIBS += $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

## Object files to link from the top build directory

LIBS += ../DDASFitHitUnpacker.o ../Configuration.o ../functions_analytic.o ../functions_template.o ../CRingItemProcessor.o

HEADERS += QTraceView.h QHitData.h QRootCanvas.h TraceViewProcessor.h DDASDecoder.h FitManager.h
SOURCES += traceview.cpp QTraceView.cpp QHitData.cpp QRootCanvas.cpp TraceViewProcessor.cpp DDASDecoder.cpp FitManager.cpp
