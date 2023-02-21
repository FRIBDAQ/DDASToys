/** 
 * @file  QTraceView.h
 * @brief Defines a Qt main applicaiton window class.
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
 *
 * Main window class for traceview responsible for management and high-level
 * control over the application. Uses Qt's signal and slot framework to
 * communicate between objects. See Qt documentation for details.
 */

class QTraceView : public QWidget
{
  Q_OBJECT

public:
  QTraceView(QWidget* parent = nullptr);
  virtual ~QTraceView();
  
private:
  virtual void changeEvent(QEvent* e);  // Overridden from base class

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
			   
private slots:
  void openFile();
  void getNextEvent();
  void skipEvents();
  void filterHits();
  void updateSelectableHits();
  void processHit();
  void handleRootEvents();
  void test();
  
private:
  
  // Our member data
  
  DDASDecoder* m_pDecoder;
  FitManager* m_pFitManager;

  bool m_config;
  bool m_templateConfig;
  std::string m_fileName;
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
