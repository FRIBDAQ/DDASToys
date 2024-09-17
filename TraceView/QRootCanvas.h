/** 
 * @file  QRootCanvas.h
 * @brief Defines a class for embedding a ROOT canvas in a Qt application.
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
 * @brief A ROOT canvas embedded in a Qt application.
 *
 * Embedded ROOT canvas in a Qt application. Overrides Qt event handling for 
 * resize and paint events (e.g. re-draw after the canvas is hidden behind 
 * another window) as well as mouse events. This allows us to manipulate the
 * ROOT canvas as expected: zooming on axes, right click actions, etc. 
 * Drawing on the canvas is bog-standard ROOT. 
 */
  
public:
    /**
     * @brief Constructor.
     * @param pFitMgr Pointer to a FitManager for plotting fit data. 
     *   The FitManager object is handled by the caller.
     * @param parent Pointer to QWidget parent object (optional, 
     *   default=nullptr).
     */
    QRootCanvas(FitManager* pFitMgr, QWidget* parent = nullptr);
    /** @brief Destructor. */
    virtual ~QRootCanvas();

    /**
     * @brief Return a pointer to the ROOT canvas.
     * @return Pointer to the ROOT canvas object.
     */
    TCanvas* getCanvas() {return m_pCanvas;};
    /**
     * @brief Draw hit data on the canvas.
     * @param hit References the hit we're plotting data from
     */
    void drawHit(const DAQ::DDAS::DDASFitHit& hit);
    /** @brief Clear and update the ROOT canvas. */
    void clear();
  
protected:

    // Qt actions to ROOT

    /**
     * @brief Handle Qt mouse move events on the ROOT canvas.
     * @param e Pointer to the QMouseEvent to handle
     */
    virtual void mouseMoveEvent(QMouseEvent* e);
    /**
     * @brief Handle Qt mouse press events on the ROOT canvas.
     * @param e Pointer to the QMouseEvent to handle
     */
    virtual void mousePressEvent(QMouseEvent* e);
    /**
     * @brief Handle Qt mouse release events on the ROOT canvas.
     * @param e Pointer to the QMouseEvent to handle
     */
    virtual void mouseReleaseEvent(QMouseEvent* e);
    /**
     * @brief Handle resize events in ROOT. */
    virtual void paintEvent(QPaintEvent*);
    /**
     * @brief Handle paint events in ROOT. */
    virtual void resizeEvent(QResizeEvent*);

private:
    /**
     * @brief Draw a trace on the ROOT canvas. 
     * @param hit References the hit we extract and plot the trace from.
     */
    void drawTrace(const DAQ::DDAS::DDASFitHit& hit);
    /**
     * @brief Draw fit data for a single pulse fit on the current axes. 
     * @param hit References the hit with fit parameters stored in 
     *   a HitExtension appended to the end of the "normal" hit.
     */
    void drawSingleFit(const DAQ::DDAS::DDASFitHit& hit);
    /**
     * @brief Draw fit data for a double pulse fit on the current axes. 
     * @param hit References the hit with fit parameters stored in 
     *   a HitExtension appended to the end of the "normal" hit.
     */
    void drawDoubleFit(const DAQ::DDAS::DDASFitHit& hit);
    /** @brief Draw a legend for the trace fits on the canvas. */
    void drawFitLegend(); 
  
private:
    FitManager* m_pFitManager; //!< Calculate trace fits (managed by caller).
    TCanvas* m_pCanvas;        //!< The ROOT canvas we draw on.
    TLegend* m_pFitLegend;     //!< Legend to plot on the canvas for each hit.
    TH1D* m_pTraceHist;        //!< Histogram of trace data.
    // Machinery to make these TF1s is irritating and besides for template
    // fitting the "fit" is itself actually binned!
    TH1D* m_pFit1Hist; //!< Histogram of single-pluse fit data.
    TH1D* m_pFit2Hist; //!< Histogram of double-pulse fit data.
};

#endif
