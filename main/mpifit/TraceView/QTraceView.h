/** 
 * @file  QTraceView.h
 * @brief Defines a Qt main applicaiton window class.
 */

/** @addtogroup traceview
 * @{
 */

#ifndef QTRACEVIEW_H
#define QTRACEVIEW_H

#include <QWidget>

#include <string>
#include <vector>

class QMenuBar;
class QMenu;
class QAction;
class QGroupBox;
class QStatusBar;
class QPushButton;
class QComboBox;
class QLineEdit;
class QListView;
class QStandardItemModel;
class QTimer;
class QCommandLineParser;
class QString;
// class QCloseEvent;

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class DDASDecoder;
class FitManager;
class QHitData;
class QRootCanvas;

/**
 * @class QTraceView
 * @brief Main traceview GUI window.
 *
 * Main window class for traceview responsible for management and high-level
 * control over the application. Uses Qt's signal and slot framework to
 * communicate between objects. See Qt documentation for details.
 */

class QTraceView : public QWidget
{
  Q_OBJECT

public:
  QTraceView(QCommandLineParser& parser, QWidget* parent = nullptr);
  virtual ~QTraceView();

  // QEvent handlers overridden from the base class.
  
protected:
  virtual void changeEvent(QEvent* e);
  // virtual void closeEvent(QCloseEvent* e);

private:
  void createActions();
  void configureMenu();
  QWidget* createTopBoxes();
  QListView* createHitSelectList();
  QWidget* createPlotWidget();
  void createConnections();

  void setStatusBar(std::string msg);
  bool isValidHit(const DAQ::DDAS::DDASFitHit& hit);
  void displayHitData(const DAQ::DDAS::DDASFitHit& hit);
  void resetGUI();
  void enableAll();
  void disableAll();
  void parseArgs(QCommandLineParser& parser);
			   
private slots:
  void openFile();
  void configureSource(QString filename);
  void getNextEvent();
  void skipEvents();
  void filterHits();
  void updateSelectableHits();
  void processHit();
  void handleRootEvents();
  void issueWarning(std::string msg);
  void test();
  
private:
  
  // Our member data
  
  DDASDecoder* m_pDecoder;
  FitManager* m_pFitManager;

  bool m_config;
  bool m_templateConfig;
  std::vector<DAQ::DDAS::DDASFitHit> m_hits;
  std::vector<DAQ::DDAS::DDASFitHit> m_filteredHits;
  
  // Added to this widget, Qt _should_ handle cleanup on destruction
  
  QMenuBar* m_pMenuBar;
  QMenu* m_pFileMenu;
  QAction* m_pOpenAction;
  QAction* m_pExitAction;

  QPushButton* m_pButtons[3];
  QPushButton* m_pSkipEvents;
  QLineEdit* m_pEventsToSkip;
  QLineEdit* m_pHitFilter[3];
  QWidget* m_pTopBoxes;

  QHitData* m_pHitData;
  
  QListView* m_pHitSelectList;
  QRootCanvas* m_pRootCanvas;
  QTimer* m_pTimer;
  
  QStatusBar* m_pStatusBar;
};

#endif

/** @} */
