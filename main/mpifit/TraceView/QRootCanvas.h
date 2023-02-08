#ifndef QROOTCANVAS_H
#define QROOTCANVAS_H

#include <QWidget>

#include <vector>
#include <cstdint>

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class TCanvas;
class TH1D;

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;

class FitManager;

class QRootCanvas : public QWidget
{
  Q_OBJECT
  
public:
  QRootCanvas(FitManager* pFitMgr, QWidget* parent = nullptr);
  virtual ~QRootCanvas();
  
  TCanvas* getCanvas() {return m_pCanvas;};
  void drawEvent(DAQ::DDAS::DDASFitHit& hit);
  void clear();

  // Qt actions to Root
protected:
  virtual void mouseMoveEvent(QMouseEvent* e);
  virtual void mousePressEvent(QMouseEvent* e);
  virtual void mouseReleaseEvent(QMouseEvent* e);
  virtual void paintEvent(QPaintEvent* e);
  virtual void resizeEvent(QResizeEvent* e);

private:
  void drawTrace(DAQ::DDAS::DDASFitHit& hit);
  void drawSingleFit(DAQ::DDAS::DDASFitHit& hit);
  void drawDoubleFit(DAQ::DDAS::DDASFitHit& hit);
  
private:
  FitManager* m_pFitManager;
  TCanvas* m_pCanvas;
  TH1D* m_pTraceHist;
  TH1D* m_pFit1Hist;
  TH1D* m_pFit2Hist;
};

#endif
