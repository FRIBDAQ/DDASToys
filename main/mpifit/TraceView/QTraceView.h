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
class Configuration;
class QHitData;
class QRootCanvas;
class FitManager;

class QTraceView : public QWidget
{
  Q_OBJECT

public:
  QTraceView(QWidget* parent = nullptr);
  virtual ~QTraceView();

  // Overridden from base class
private:
  virtual void changeEvent(QEvent* e);

private:
  QGroupBox* createTopGroupBox();
  QListView* createHitSelectList();
  QWidget* createPlotWidget();  
  void createConnections();
  void createActions();
  void configureMenu();

  void setStatusBar(std::string msg);
  bool isValidHit(const DAQ::DDAS::DDASFitHit& hit);
  void displayHitData(const DAQ::DDAS::DDASFitHit& hit);
  void enableAll();
  void disableAll();
			   
private slots:
  void openFile();
  void getNextEvent();
  void filterHits();
  void updateSelectableHits();
  void processHit();
  void handleRootEvents();
  void test();
  
private:
  // Our member data
  DDASDecoder* m_pDecoder;
  FitManager* m_pFitManager;

  int m_count;
  bool m_config;
  bool m_templateConfig;
  std::string m_fileName;
  std::vector<DAQ::DDAS::DDASFitHit> m_hits;
  std::vector<DAQ::DDAS::DDASFitHit> m_filteredHits;
  
  // Added to the top widget, Qt _should_ handle cleanup
  QMenuBar* m_pMenuBar;
  QMenu* m_pFileMenu;
  QAction* m_pOpenAction;
  QAction* m_pExitAction;

  QPushButton* m_pButtons[3];
  QLineEdit* m_pHitFilter[3];
  QGroupBox* m_pTopGroupBox;

  QHitData* m_pHitData;
  
  QListView* m_pHitSelectList;
  QRootCanvas* m_pRootCanvas;
  QTimer* m_pTimer;
  
  QStatusBar* m_pStatusBar;
};

#endif
