#include <cstdlib>

#include <QApplication>

#include <TApplication.h>
#include <TSystem.h>

#include "QTraceView.h"

int main(int argc, char *argv[])
{
  TApplication rootapp("Simple Qt ROOT Application", &argc, argv);
  QApplication app(argc, argv);

  QTraceView window(nullptr);
  window.setWindowTitle("DDAS TraceView");
  window.resize(window.sizeHint());
  window.show();
  
  QObject::connect(qApp, SIGNAL(lastWindowClosed()), qApp, SLOT(quit()));

  return app.exec();
}
