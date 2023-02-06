#ifndef QHITDATA_H
#define QHITDATA_H

#include <QGroupBox>

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}
class QLabel;

class QHitData : public QGroupBox
{
  Q_OBJECT

public:
  QHitData();
  ~QHitData();

  void update(const DAQ::DDAS::DDASFitHit& hit);

private:
  void updateRawData(const DAQ::DDAS::DDASFitHit& hit);
  
private:
  struct RawHitData
  {
    QLabel* s_pId;
    QLabel* s_pRawData;
  } m_rawHitData;
  
};

#endif
