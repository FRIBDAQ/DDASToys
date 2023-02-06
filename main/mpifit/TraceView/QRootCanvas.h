#ifndef QROOTCANVAS_H
#define QROOTCANVAS_H

#include <QWidget>

#include <vector>
#include <cstdint>

class TCanvas;
class TH1D;

class QMouseEvent;
class QPaintEvent;
class QResizeEvent;

class QRootCanvas : public QWidget
{
  Q_OBJECT
  
public:
  QRootCanvas(QWidget* parent = nullptr);
  virtual ~QRootCanvas();
  
  TCanvas* getCanvas() {return m_pCanvas;};
  void DrawTrace(std::vector<std::uint16_t>& trace);
  void Clear();
  
protected:
  virtual void mouseMoveEvent(QMouseEvent* e);
  virtual void mousePressEvent(QMouseEvent* e);
  virtual void mouseReleaseEvent(QMouseEvent* e);
  virtual void paintEvent(QPaintEvent* e);
  virtual void resizeEvent(QResizeEvent* e);
   
private:
  TCanvas* m_pCanvas;
  TH1D* m_pTraceHist;
};

#endif
