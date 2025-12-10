TEMPLATE = app
  
QT += widgets

QMAKE_CXXFLAGS += $(shell $(ROOTSYS)/bin/root-config --cflags)
CONFIG += qt warn_on thread console

message(Using DAQROOT="$(DAQROOT)")
message(Using DDASFMTINC="$$DDASFMTINC")
message(Using DDASFMTLIB="$$DDASFMTLIB")
message(Using UFMTROOT="$$UFMTROOT")
message(Using ROOTSYS="$(ROOTSYS)")

INCLUDEPATH += .. $${DDASFMTINC} $(DAQINC) $${UFMTROOT}/include $(ROOTSYS)/include

## FRIBDAQ libraries

LIBS += -L$${DDASFMTLIB} -lDDASFormat -Wl,-rpath=$${DDASFMTLIB} -L$(DAQLIB) -ldataformat -ldaqio -lException -lurl -Wl,-rpath=$(DAQLIB) -L$${UFMTROOT}/lib -lAbstractFormat -Wl,-rpath=$${UFMTROOT}/lib

## ROOT libraries

LIBS += $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

## Object files to link from the top build directory

LIBS += ../DDASFitHitUnpacker.o ../Configuration.o ../functions_analytic.o ../functions_template.o ../CRingItemProcessor.o

HEADERS += QTraceView.h QHitData.h QRootCanvas.h TraceViewProcessor.h DDASDecoder.h FitManager.h
SOURCES += traceview.cpp QTraceView.cpp QHitData.cpp QRootCanvas.cpp TraceViewProcessor.cpp DDASDecoder.cpp FitManager.cpp
