/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  QRootCanvas.cpp
 * @brief Implementation of Qt-embedded ROOT canvas class
 */

#include "QRootCanvas.h"

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

using namespace ddastoys;

//____________________________________________________________________________
/**
 * @details
 * Construct a QWidget and register it with the ROOT graphics backend.
 * The canvas does not own the FitManager object.
 */
QRootCanvas::QRootCanvas(FitManager* pFitMgr, QWidget* parent) :
    QWidget(parent), m_pFitManager(pFitMgr), m_pCanvas(nullptr),
    m_pFitLegend(nullptr), m_pTraceHist(nullptr), m_pFit1Hist(nullptr),
    m_pFit2Hist(nullptr)
{
    // Set options needed to properly update the canvas when resizing the
    // widget and to properly handle context menus and mouse move events.
  
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setMinimumSize(800, 600);
    setUpdatesEnabled(kFALSE);
    setMouseTracking(kTRUE);

    // Register the QWidget in TVirtualX, giving its native window id.
  
    int wid = gVirtualX->AddWindow((ULong_t)winId(), 800, 600);

    // Create the ROOT TCanvas, giving as argument the QWidget registered id.
  
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
 * @brief Destructor.
 */
QRootCanvas::~QRootCanvas()
{
    delete m_pCanvas;
    delete m_pFitLegend;
    delete m_pTraceHist;
    delete m_pFit1Hist;
    delete m_pFit2Hist;
}

///
// Public methods
// 

//____________________________________________________________________________
void
QRootCanvas::drawHit(const DDASFitHit& hit)
{  
    drawTrace(hit);

    // For single pulses the double fit and single fit frequently look almost
    // identical. Drawing the double fit first makes the single fit in these
    // instances appear more clear.
  
    if (hit.hasExtension()) {
	drawDoubleFit(hit);
	drawSingleFit(hit);
	drawFitLegend();
    }

  
    m_pCanvas->Modified();
    m_pCanvas->Update();
}

//____________________________________________________________________________
void
QRootCanvas::clear()
{
    if (m_pCanvas) {
	m_pCanvas->Clear();
	m_pCanvas->Modified();
	m_pCanvas->Update();
    }
}

///
// Protected methods
//

//____________________________________________________________________________
void
QRootCanvas::mouseMoveEvent(QMouseEvent *e)
{
    if (m_pCanvas) {
	if (e->buttons() & Qt::LeftButton) {
	    m_pCanvas->HandleInput(kButton1Motion, e->x(), e->y());
	} else if (e->buttons() & Qt::MiddleButton) {
	    m_pCanvas->HandleInput(kButton2Motion, e->x(), e->y());
	} else if (e->buttons() & Qt::RightButton) {
	    m_pCanvas->HandleInput(kButton3Motion, e->x(), e->y());
	} else {
	    m_pCanvas->HandleInput(kMouseMotion, e->x(), e->y());
	}
    }
}

//____________________________________________________________________________
void
QRootCanvas::mousePressEvent(QMouseEvent *e)
{
    if (m_pCanvas) {
	switch (e->button()) {
	case Qt::LeftButton :
	    m_pCanvas->HandleInput(kButton1Down, e->x(), e->y());
	    break;
	case Qt::MiddleButton :
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
void
QRootCanvas::mouseReleaseEvent(QMouseEvent *e)
{
    if (m_pCanvas) {
	switch (e->button()) {
	case Qt::LeftButton :
	    m_pCanvas->HandleInput(kButton1Up, e->x(), e->y());
	    break;
	case Qt::MiddleButton :
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
void
QRootCanvas::resizeEvent(QResizeEvent*)
{
    if (m_pCanvas) {
	m_pCanvas->Resize();
	m_pCanvas->Update();
    }
}

//____________________________________________________________________________
void
QRootCanvas::paintEvent(QPaintEvent*)
{
    if (m_pCanvas) {
	m_pCanvas->Resize();
	m_pCanvas->Update();
    }
}

///
// Private methods
//

//____________________________________________________________________________
/**
 * @brief Draw a trace on the ROOT canvas. 
 * 
 * Expects a trace of uint16_t from a DDAS hit.
 *
 * @param hit  References the hit we extract and plot the trace from
 */
void
QRootCanvas::drawTrace(const DDASFitHit& hit)
{
    std::vector<uint16_t> trace = hit.getTrace();
  
    // Create histograms if they do not exist, otherwise ensure correct size.
  
    if (!m_pTraceHist) {
	m_pTraceHist = new TH1D(
	    "trace", "trace", trace.size(), 0, trace.size()
	    );
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
 * @details
 * Note that this function expects that the hit has a fit extension which must
 * be checked by the caller using hit.hasExtension().
 */
void
QRootCanvas::drawSingleFit(const DDASFitHit& hit)
{  
    unsigned low = m_pFitManager->getLowFitLimit(hit);
    unsigned high = m_pFitManager->getHighFitLimit(hit);
    unsigned fitRange = high - low + 1; // +1 since range is [low, high]

    // Get a vector of fit data with a size defined by the fitting range.
    // Note that index 0 of the fit vector corresponds to sample number low
    // on the actual trace. We assume if we are here, we have an extension.
  
    std::vector<double> fit = m_pFitManager->getSinglePulseFit(hit, low, high);
  
    if (!m_pFit1Hist) {
	m_pFit1Hist = new TH1D("fit1", "fit1", fitRange, low, high+1);
    } else {
	m_pFit1Hist->SetBins(fitRange, low, high+1);
    }
    m_pFit1Hist->Reset("ICESM");
  
    for (unsigned i = low; i <= high; i++) {
	m_pFit1Hist->Fill(i, fit[i-low]);
    }
    m_pFit1Hist->SetLineColor(kRed);
    std::string options = "hist same";
    auto method = m_pFitManager->getMethod();
    if (method == ANALYTIC || method == ML_INFERENCE) {
	options = "hist c same";
    }
    m_pFit1Hist->Draw(options.c_str());
}

//____________________________________________________________________________
/**
 * @details
 * Note that this function expects that the hit has a fit extension which must 
 * be checked by the caller using hit.hasExtension().
 */
void
QRootCanvas::drawDoubleFit(const DDASFitHit& hit)
{ 
    unsigned low = m_pFitManager->getLowFitLimit(hit);
    unsigned high = m_pFitManager->getHighFitLimit(hit);
    unsigned fitRange = high - low + 1; // +1 since range is [low, high]

    // Get a vector of fit data with a size defined by the fitting range.
    // Note that index 0 of the fit vector corresponds to sample number low
    // on the actual trace. We assume if we are here, we have an extension.
  
    std::vector<double> fit = m_pFitManager->getDoublePulseFit(hit, low, high);
  
    if (!m_pFit2Hist) {
	m_pFit2Hist = new TH1D("fit2", "fit2", fitRange, low, high+1);
    } else {
	m_pFit2Hist->SetBins(fitRange, low, high+1);
    }
    m_pFit2Hist->Reset("ICESM");
  
    for (unsigned i = low; i <= high; i++) {
	m_pFit2Hist->Fill(i, fit[i-low]);
    }
    m_pFit2Hist->SetLineColor(kBlue);
    std::string options = "hist same";
    auto method = m_pFitManager->getMethod();
    if (method == ANALYTIC || method == ML_INFERENCE) {
	options = "hist c same";
    }
    m_pFit2Hist->Draw(options.c_str());
}

//____________________________________________________________________________
void
QRootCanvas::drawFitLegend()
{
    if (!m_pFitLegend) {
	m_pFitLegend = new TLegend(0.7, 0.8, 0.95, 0.95);
	m_pFitLegend->SetBorderSize(0);      
	m_pFitLegend->AddEntry(m_pFit1Hist, "Single pulse fit");
	m_pFitLegend->AddEntry(m_pFit2Hist, "Double pulse fit");
    }
  
    m_pFitLegend->Draw();  
}
