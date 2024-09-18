/** 
 * @file  traceview.cpp
 * @brief traceview main. Run the Qt application.
 */

#include <cstdlib>

#include <QApplication>
#include <QCommandLineParser>

#include <TApplication.h>
#include <TSystem.h>

#include "QTraceView.h"

/**
 * @brief traceview main.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 *
 * @return int
 * @retval 0   Application exits successfully.
 * @retval !=0 Error.
 *
 * @details
 * Create the main application. The application has a command line parser to 
 * allow the user to pass commands at runtime to configure the initial viewer 
 * settings. A TApplication necessary for the embedded Root canvas is also 
 * created.
 */
int main(int argc, char* argv[])
{
  QApplication app(argc, argv);
  app.setApplicationName("traceview");
  app.setApplicationVersion("1.0");

  // Note: "any option value that looks like a builtin Qt option, will be
  // treated by QCoreApplication as a builtin Qt option," see
  // https://doc.qt.io/qt-5/qcommandlineparser.html. Default builtin options are
  // listed here: // https://doc.qt.io/qt-5/qapplication.html#QApplication and
  // https://doc.qt.io/qt-5/qguiapplication.html#supported-command-line-option.
  // This may not be an exhaustive list.

   QCommandLineParser parser;
   parser.setApplicationDescription("traceview command line parser");
   parser.addHelpOption();
   parser.addVersionOption();
   parser.addOptions({
	  {{"s", "source"}, "Data source file (.evt).", "source"},
	  {{"m", "method"}, "Trace fitting method.", "method", "template"}
     });
   parser.process(app);

   // After the QCommandLineParser so ROOT TApplication parser doesn't consume
   // shared default args such as --help and --version.   
   TApplication rootapp("Simple Qt ROOT Application", &argc, argv);

   // Display the GUI
  
   ddastoys::QTraceView window(parser, nullptr);
   window.setWindowTitle("DDAS TraceView");
   window.resize(window.sizeHint());
   window.show();
  
   QObject::connect(qApp, SIGNAL(lastWindowClosed()), qApp, SLOT(quit()));
   
   return app.exec();
}
