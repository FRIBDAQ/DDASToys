#include "QRootCanvas.h"

#include <TCanvas.h>
#include <TVirtualX.h>
#include <TH1D.h>
#include <TStyle.h>

#include <QMouseEvent>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QEvent>
#include <QTimer>

QRootCanvas::QRootCanvas(QWidget* parent) :
  QWidget(parent),
  m_pCanvas(nullptr),
  m_pTraceHist(nullptr)
{
  // Set options needed to properly update the canvas when resizing the widget
  // and to properly handle context menus and mouse move events
  setAttribute(Qt::WA_OpaquePaintEvent, true);
  setMinimumSize(800, 600);
  setUpdatesEnabled(kFALSE);
  setMouseTracking(kTRUE);

  // Register the QWidget in TVirtualX, giving its native window id
  int wid = gVirtualX->AddWindow((ULong_t)winId(), 800, 600);

  // Create the Root TCanvas, giving as argument the QWidget registered id
  m_pCanvas = new TCanvas("c1", width(), height(), wid);
  
  m_pCanvas->SetLeftMargin(0.125);
  m_pCanvas->SetRightMargin(0.025);
  m_pCanvas->SetTopMargin(0.025);
  m_pCanvas->SetBottomMargin(0.125);
  
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
}

QRootCanvas::~QRootCanvas()
{
  delete m_pCanvas;
  delete m_pTraceHist;
}

//
// Protected methods
//

void
QRootCanvas::mouseMoveEvent(QMouseEvent *e)
{
  if (m_pCanvas) {
    if (e->buttons() & Qt::LeftButton) {
      m_pCanvas->HandleInput(kButton1Motion, e->x(), e->y());
    } else if (e->buttons() & Qt::MidButton) {
      m_pCanvas->HandleInput(kButton2Motion, e->x(), e->y());
    } else if (e->buttons() & Qt::RightButton) {
      m_pCanvas->HandleInput(kButton3Motion, e->x(), e->y());
    } else {
      m_pCanvas->HandleInput(kMouseMotion, e->x(), e->y());
    }
  }
}

void
QRootCanvas::mousePressEvent(QMouseEvent *e)
{
  if (m_pCanvas) {
    switch (e->button()) {
    case Qt::LeftButton :
      m_pCanvas->HandleInput(kButton1Down, e->x(), e->y());
      break;
    case Qt::MidButton :
      m_pCanvas->HandleInput(kButton2Down, e->x(), e->y());
      break;
    case Qt::RightButton :
      m_pCanvas->HandleInput(kButton3Down, e->x(), e->y());
      break;
    default:
      break;
    }
  }
}

void
QRootCanvas::mouseReleaseEvent(QMouseEvent *e)
{
  if (m_pCanvas) {
    switch (e->button()) {
    case Qt::LeftButton :
      m_pCanvas->HandleInput(kButton1Up, e->x(), e->y());
      break;
    case Qt::MidButton :
      m_pCanvas->HandleInput(kButton2Up, e->x(), e->y());
      break;
    case Qt::RightButton :
      m_pCanvas->HandleInput(kButton3Up, e->x(), e->y());
      break;
    default:
      break;
    }
  }
}

void
QRootCanvas::resizeEvent(QResizeEvent*)
{
  if (m_pCanvas) {
    m_pCanvas->Resize();
    m_pCanvas->Update();
  }
}

void
QRootCanvas::paintEvent(QPaintEvent*)
{
  if (m_pCanvas) {
    m_pCanvas->Resize();
    m_pCanvas->Update();
  }
}

void
QRootCanvas::DrawTrace(std::vector<std::uint16_t>& trace)
{
  // Create the trace histogram if it doesn't exist, otherwise just ensure
  // it has the correct binning
  if (!m_pTraceHist) {
    m_pTraceHist = new TH1D("trace", "trace", trace.size(), 0, trace.size());
  } else {
    m_pTraceHist->SetBins(trace.size(), 0, trace.size());
  }
  m_pTraceHist->Reset("ICESM");
  
  for (unsigned i=0; i<trace.size(); i++) {
    m_pTraceHist->Fill(i, trace[i]);
  }
  m_pTraceHist->SetLineColor(kBlack);
  m_pTraceHist->GetXaxis()->SetTitle("Sample number");
  m_pTraceHist->GetYaxis()->SetTitle("ADC value [arb.]");
  m_pTraceHist->Draw("hist");

  m_pCanvas->Modified();
  m_pCanvas->Update();
}

void
QRootCanvas::Clear()
{
  if (m_pCanvas) {
    m_pCanvas->Clear();
    m_pCanvas->Modified();
    m_pCanvas->Update();
  }
}
