/** @file: QRootCanvas.h
 *  @brief: Defines a class for embedding a Root canvas in a Qt application.
 */

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

/**
 * @class QRootCanvas
 *
 *   Embedded Root canvas in a Qt application. Overrides Qt event handling for 
 *   resize and paint events (e.g. re-draw after the canvas is hidden behind 
 *   another window) as well as mouse events. This allows us to manipulate the
 *   Root canvas as expected: zooming on axes, right click actions, etc. 
 *   Drawing on the canvas is bog-standard Root. 
 */
  
public:
  QRootCanvas(FitManager* pFitMgr, QWidget* parent = nullptr);
  virtual ~QRootCanvas();
  
  TCanvas* getCanvas() {return m_pCanvas;};
  void drawHit(const DAQ::DDAS::DDASFitHit& hit);
  void clear();

  // Qt actions to Root
protected:
  virtual void mouseMoveEvent(QMouseEvent* e);
  virtual void mousePressEvent(QMouseEvent* e);
  virtual void mouseReleaseEvent(QMouseEvent* e);
  virtual void paintEvent(QPaintEvent* e);
  virtual void resizeEvent(QResizeEvent* e);

private:
  void drawTrace(const DAQ::DDAS::DDASFitHit& hit);
  void drawSingleFit(const DAQ::DDAS::DDASFitHit& hit);
  void drawDoubleFit(const DAQ::DDAS::DDASFitHit& hit);
  
private:
  FitManager* m_pFitManager;
  TCanvas* m_pCanvas;
  TH1D* m_pTraceHist;
  TH1D* m_pFit1Hist;
  TH1D* m_pFit2Hist;
};

#endif
