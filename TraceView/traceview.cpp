/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

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

using namespace ddastoys;

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
  app.setApplicationVersion("2.0");

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

   // Set options:
   
   std::vector<QCommandLineOption> opts;
   QCommandLineOption sourceOpt(
       QStringList() << "s" << "source",
       QCoreApplication::translate("main", "Path to input data file (.evt)"),
       QCoreApplication::translate("main", "source")
       );
   opts.push_back(sourceOpt);
   QCommandLineOption methodOpt(
       QStringList() << "m" << "method",
       QCoreApplication::translate(
	   "main",
	   "Fitting method (one of 'analytic', 'template', 'ml_inference')"
	   ),
       QCoreApplication::translate("main", "method", "a")
       );
   opts.push_back(methodOpt);

   // Add options and configure the parser:
   
   for (const auto& opt : opts) {
       parser.addOption(opt);
   }
   
   parser.process(app);

   // After the QCommandLineParser so ROOT TApplication parser doesn't consume
   // shared default args such as --help and --version.   
   TApplication rootapp("Simple Qt ROOT Application", &argc, argv);

   // Display the GUI
  
   QTraceView window(parser, nullptr);
   window.setWindowTitle("DDAS TraceView");
   window.resize(window.sizeHint());
   window.show();
  
   QObject::connect(qApp, SIGNAL(lastWindowClosed()), qApp, SLOT(quit()));
   
   return app.exec();
}
