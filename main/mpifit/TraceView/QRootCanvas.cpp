/** @file: QRootCanvas.cpp
 *  @breif: Implementation of Qt-embedded Root canvas class
 */

#include "QRootCanvas.h"

#include <utility>
#include <iostream>

#include <TCanvas.h>
#include <TVirtualX.h>
#include <TH1D.h>
#include <TStyle.h>
#include <TLegend.h>

#include <QMouseEvent>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QEvent>
#include <QTimer>

#include <Configuration.h>
#include <DDASFitHit.h>
#include "FitManager.h"

//____________________________________________________________________________
/**
 * Constructor
 *   Create QWidget and register it with the Root graphics backend.
 *
 * @param pFitMgr - pointer to a FitManager for plotting fit data. The 
 *                  FitManager object is handled by the caller.
 * @param parent  - pointer to QWidget parent object, default = nullptr
 */
QRootCanvas::QRootCanvas(FitManager* pFitMgr, QWidget* parent) :
  QWidget(parent), m_pFitManager(pFitMgr), m_pCanvas(nullptr),
  m_pLegend(nullptr), m_pTraceHist(nullptr), m_pFit1Hist(nullptr),
  m_pFit2Hist(nullptr)
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
  m_pCanvas = new TCanvas("RootCanvas", width(), height(), wid);
  
  m_pCanvas->SetLeftMargin(0.125);
  m_pCanvas->SetRightMargin(0.025);
  m_pCanvas->SetTopMargin(0.025);
  m_pCanvas->SetBottomMargin(0.125);
  
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
}

//____________________________________________________________________________
/**
 * Destructor
 */
QRootCanvas::~QRootCanvas()
{
  delete m_pCanvas;
  delete m_pLegend;
  delete m_pTraceHist;
  delete m_pFit1Hist;
  delete m_pFit2Hist;
}

//
// Public methods
// 

//____________________________________________________________________________
/**
 * drawHit
 *   Draw hit data on the canvas.
 * 
 * @param hit - references the hit we're plotting data from
 */
void
QRootCanvas::drawHit(const DAQ::DDAS::DDASFitHit& hit)
{  
  drawTrace(hit);

  // For single pulses the double fit and single fit frequently look almost
  // identical. Drawing the double fit first makes the single fit in these
  // instances appear more clear.
  if (hit.hasExtension()) {
    drawDoubleFit(hit);
    drawSingleFit(hit);
  }

  drawLegend(hit.hasExtension());
  
  m_pCanvas->Modified();
  m_pCanvas->Update();
}

//____________________________________________________________________________
/**
 * clear
 *   Clear and update the Root canvas.
 */
void
QRootCanvas::clear()
{
  if (m_pCanvas) {
    m_pCanvas->Clear();
    m_pCanvas->Modified();
    m_pCanvas->Update();
  }
}

//
// Protected methods
//

//____________________________________________________________________________
/**
 * mouseMoveEvent
 *   Handle Qt mouse move events on the Root canvas.
 *
 * @param e - pointer to the QMouseEvent to handle
 */
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

//____________________________________________________________________________
/**
 * mousePressEvent
 *   Handle Qt mouse press events on the Root canvas.
 *
 * @param e - pointer to the QMouseEvent to handle
 */
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

//____________________________________________________________________________
/**
 * mouseReleaseEvent
 *   Handle Qt mouse release events on the Root canvas.
 *
 * @param e - pointer to the QMouseEvent to handle
 */
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

//____________________________________________________________________________
/**
 * resizeEvent
 *   Handle resize events in Root. See Qt documentation for details.
 *
 * @param - pointer to resize event
 */
void
QRootCanvas::resizeEvent(QResizeEvent*)
{
  if (m_pCanvas) {
    m_pCanvas->Resize();
    m_pCanvas->Update();
  }
}

//____________________________________________________________________________
/**
 * paintEvent
 *   Handle paint events in Root. See Qt documentaiton for details.
 *
 * @param - pointer to paint event
 */
void
QRootCanvas::paintEvent(QPaintEvent*)
{
  if (m_pCanvas) {
    m_pCanvas->Resize();
    m_pCanvas->Update();
  }
}

//
// Private methods
//

//____________________________________________________________________________
/**
 * drawTrace
 *   Draw a trace on the Root canvas. Expects a trace of uint16_t from a 
 *   DDAS hit.
 *
 * @param hit - references the hit we extract and plot the trace from
 */
void
QRootCanvas::drawTrace(const DAQ::DDAS::DDASFitHit& hit)
{
  std::vector<std::uint16_t> trace = hit.GetTrace();
  
  // Create histograms if they do not exist, otherwise ensure correct size
  if (!m_pTraceHist) {
    m_pTraceHist = new TH1D("trace", "trace", trace.size(), 0, trace.size());
  } else {
    m_pTraceHist->SetBins(trace.size(), 0, trace.size());
  }
  m_pTraceHist->Reset("ICESM");
  
  // Add to canvas and configure display options. Since we always draw a
  // trace, the axis labels are set here, and subsequent plots are drawn
  // on these axes.
  for (unsigned i=0; i<trace.size(); i++) {
    m_pTraceHist->Fill(i, trace[i]);
  }
  m_pTraceHist->SetLineColor(kBlack);
  m_pTraceHist->GetXaxis()->SetTitle("Sample number");
  m_pTraceHist->GetYaxis()->SetTitle("ADC value [arb.]");
  m_pTraceHist->Draw("hist");
}

//____________________________________________________________________________
/**
 * drawSingleFit
 *   Draw fit data for a single pulse fit on the current axes. Note that this 
 *   function expects that the hit has a fit extension which must be checked
 *   by the caller using hit.hasExtension().
 * 
 * @param hit - references the hit containing with fit parameters stored in 
 *              a HitExtension appended to the end of the "normal" hit
 */
void
QRootCanvas::drawSingleFit(const DAQ::DDAS::DDASFitHit& hit)
{  
  unsigned low = m_pFitManager->getLowFitLimit(hit);
  unsigned high = m_pFitManager->getHighFitLimit(hit);
  unsigned fitRange = high - low;

  // Get a vector of fit data with a size defined by the fitting range.
  // Note that index 0 of the fit vector corresponds to sample number low
  // on the actual trace. We assume if we are here, we have an extension.
  DDAS::HitExtension ext = hit.getExtension();
  std::vector<double> fit = m_pFitManager->getSinglePulseFit(ext, low, high);
  
  if (!m_pFit1Hist) {
    m_pFit1Hist = new TH1D("fit1", "fit1", fitRange, low, high);
  } else {
    m_pFit1Hist->SetBins(fitRange, low, high);
  }
  m_pFit1Hist->Reset("ICESM");
  
  for (unsigned i=low; i<high; i++) {
    m_pFit1Hist->Fill(i, fit[i-low]);
  }
  m_pFit1Hist->SetLineColor(kRed);
  std::string options = "hist same";
  if (m_pFitManager->getMethod() == ANALYTIC) {
    options = "hist c same";
  }
  m_pFit1Hist->Draw(options.c_str());
}

//____________________________________________________________________________
/**
 * drawDoubleFit
 *   Draw fit data for a double pulse fit on the current axes. Note that this 
 *   function expects that the hit has a fit extension which must be checked
 *   by the caller using hit.hasExtension().
 *
 * @param hit - references the hit containing with fit parameters stored in 
 *              a HitExtension appended to the end of the "normal" hit
 */
void
QRootCanvas::drawDoubleFit(const DAQ::DDAS::DDASFitHit& hit)
{ 
  unsigned low = m_pFitManager->getLowFitLimit(hit);
  unsigned high = m_pFitManager->getHighFitLimit(hit);
  unsigned fitRange = high - low;

  // Get a vector of fit data with a size defined by the fitting range.
  // Note that index 0 of the fit vector corresponds to sample number low
  // on the actual trace. We assume if we are here, we have an extension.
  DDAS::HitExtension ext = hit.getExtension();
  std::vector<double> fit = m_pFitManager->getDoublePulseFit(ext, low, high);
  
  if (!m_pFit2Hist) {
    m_pFit2Hist = new TH1D("fit2", "fit2", fitRange, low, high);
  } else {
    m_pFit2Hist->SetBins(fitRange, low, high);
  }
  m_pFit2Hist->Reset("ICESM");
  
  for (unsigned i=low; i<high; i++) {
    m_pFit2Hist->Fill(i, fit[i-low]);
  }
  m_pFit2Hist->SetLineColor(kBlue);
  std::string options = "hist same";
  if (m_pFitManager->getMethod() == ANALYTIC) {
    options = "hist c same";
  }
  m_pFit2Hist->Draw(options.c_str());
}

//____________________________________________________________________________
/**
 * drawLegend
 *   Draw a legend on the canvas depending on what data exists for this hit.
 *
 * @param hasFit - true if the hit has a fit extension, false otherwise
 */
void
QRootCanvas::drawLegend(bool hasFit) {
  if (!m_pLegend) {
    m_pLegend = new TLegend(0.8, 0.8, 0.95, 0.95);
    m_pLegend->SetBorderSize(0);
  } else {
    m_pLegend->Clear();
  }

  m_pLegend->AddEntry(m_pTraceHist, "Trace");

  if (hasFit) {
    m_pLegend->AddEntry(m_pFit1Hist, "Single pulse fit");
    m_pLegend->AddEntry(m_pFit2Hist, "Double pulse fit");
  }
  
  m_pLegend->Draw();  
}
