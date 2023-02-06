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
class QHitData;
class QRootCanvas;

class QTraceView : public QWidget
{
  Q_OBJECT

public:
  QTraceView(QWidget* parent = nullptr);
  virtual ~QTraceView();

  // Overridden from base class
protected:
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

protected:
  QRootCanvas* m_pRootCanvas;
  QTimer* m_pTimer;
  
private:
  DDASDecoder* m_pDecoder;

  QMenuBar* m_pMenuBar;
  QMenu* m_pFileMenu;
  QAction* m_pOpenAction;
  QAction* m_pExitAction;

  QPushButton* m_pButtons[3];
  QLineEdit* m_pHitFilter[3];
  QGroupBox* m_pTopGroupBox;

  QHitData* m_pHitData;
  
  QListView* m_pHitSelectList;
  
  QStatusBar* m_pStatusBar;

  int m_count;
  std::string m_fileName;
  std::vector<DAQ::DDAS::DDASFitHit> m_hits;
  std::vector<DAQ::DDAS::DDASFitHit> m_filteredHits;
};

#endif
