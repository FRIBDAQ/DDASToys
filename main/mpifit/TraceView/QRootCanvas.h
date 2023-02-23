/** 
 * @file  QRootCanvas.h
 * @brief Defines a class for embedding a Root canvas in a Qt application.
 */

/** @addtogroup traceview
 * @{
 */

#ifndef QROOTCANVAS_H
#define QROOTCANVAS_H

#include <QWidget>

#include <vector>
#include <cstdint>

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;

class TCanvas;
class TH1D;
class TLegend;

namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class FitManager;

class QRootCanvas : public QWidget
{
  Q_OBJECT

/**
 * @class QRootCanvas
 * @brief A Root canvas embedded in a Qt application.
 *
 * Embedded Root canvas in a Qt application. Overrides Qt event handling for 
 * resize and paint events (e.g. re-draw after the canvas is hidden behind 
 * another window) as well as mouse events. This allows us to manipulate the
 * Root canvas as expected: zooming on axes, right click actions, etc. 
 * Drawing on the canvas is bog-standard Root. 
 */
  
public:
  QRootCanvas(FitManager* pFitMgr, QWidget* parent = nullptr);
  virtual ~QRootCanvas();

  /**
   * @brief Return a pointer to the Root canvas.
   * @return TCanvas*  Pointer to the Root canvas object.
   */
  TCanvas* getCanvas() {return m_pCanvas;};
  
  void drawHit(const DAQ::DDAS::DDASFitHit& hit);
  void clear();
  
protected:

  // Qt actions to Root
  
  virtual void mouseMoveEvent(QMouseEvent* e);
  virtual void mousePressEvent(QMouseEvent* e);
  virtual void mouseReleaseEvent(QMouseEvent* e);
  virtual void paintEvent(QPaintEvent*);
  virtual void resizeEvent(QResizeEvent*);

private:
  void drawTrace(const DAQ::DDAS::DDASFitHit& hit);
  void drawSingleFit(const DAQ::DDAS::DDASFitHit& hit);
  void drawDoubleFit(const DAQ::DDAS::DDASFitHit& hit);
  void drawFitLegend(); 
  
private:
  FitManager* m_pFitManager;
  TCanvas* m_pCanvas;
  TLegend* m_pFitLegend;
  TH1D* m_pTraceHist;
  TH1D* m_pFit1Hist;
  TH1D* m_pFit2Hist;
};

#endif

/** @} */
