/** 
 * @file  QHitData.h
 * @brief Definition of class to handle disaplying hit data on the main 
 * window UI.
 */

/** @addtogroup traceview
 * @{
 */

#ifndef QHITDATA_H
#define QHITDATA_H

#include <QWidget>

#include <string>

class QLabel;
class QComboBox;
class QPushButton;
class QGroupBox;
class QString;

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}
namespace DDAS{
  struct HitExtension;
}

class FitManager;

/**
 * @class QHitData
 * @brief Widget for managing hit information when interating with the GUI.
 *
 * Widget class for managing UI related to the hit data. Displays relavent 
 * infomration for the currently displayed hit with optional UI elements to
 * see more information e.g. a button to print the fit results to stdout.
 */

class QHitData : public QWidget
{
  Q_OBJECT

public:
  QHitData(FitManager* pFitMgr, QWidget* parent = nullptr);
  ~QHitData();

  void update(const DAQ::DDAS::DDASFitHit& hit);
  void setFitMethod(QString method);

private:
  QGroupBox* createHitBox();
  QGroupBox* createClassifierBox();
  QGroupBox* createFitBox();
  void createConnections();
  void updateHitData(const DAQ::DDAS::DDASFitHit& hit);

private slots:
  void configureFit();
  void printFitResults();
  
private:
  FitManager* m_pFitManager;
  DDAS::HitExtension* m_pExtension;
  
  QLabel* m_pId;
  QLabel* m_pRawData;
  QLabel* m_pFit1Prob;
  QLabel* m_pFit2Prob;
  QComboBox* m_pFitMethod;
  QPushButton* m_pPrintFit;
};

#endif

/** @} */
