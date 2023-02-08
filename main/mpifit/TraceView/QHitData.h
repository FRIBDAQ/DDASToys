#ifndef QHITDATA_H
#define QHITDATA_H

#include <QWidget>

#include <string>

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}
namespace DDAS{
  struct HitExtension;
}

class QLabel;
class QComboBox;
class QPushButton;
class QGroupBox;

class FitManager;

class QHitData : public QWidget
{
  Q_OBJECT

public:
  QHitData(FitManager* pFitMgr);
  ~QHitData();

  void update(DAQ::DDAS::DDASFitHit& hit);

private:
  QGroupBox* createHitBox();
  QGroupBox* createClassifierBox();
  QGroupBox* createFitBox();
  void createConnections();
  void updateHitData();

private slots:
  void configureFit();
  void printFitResults();
  
private:
  FitManager* m_pFitManager;
  DAQ::DDAS::DDASFitHit* m_pHit;
  
  QLabel* m_pId;
  QLabel* m_pRawData;
  QLabel* m_pFit1Prob;
  QLabel* m_pFit2Prob;
  QComboBox* m_pFitMethod;
  QPushButton* m_pPrintFit;
};



#endif
