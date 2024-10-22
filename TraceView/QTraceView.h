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
 * @file  QTraceView.h
 * @brief Define a Qt main applicaiton window class.
 */

#ifndef QTRACEVIEW_H
#define QTRACEVIEW_H

#include <QWidget>

#include <string>
#include <vector>

class QMenuBar;
class QMenu;
class QAction;
class QGroupBox;
class QStatusBar;
class QPushButton;
class QComboBox;
class QLineEdit;
class QListView;
class QStandardItemModel;
class QTimer;
class QCommandLineParser;
class QString;

namespace ddastoys {    
    class DDASFitHit;
}

class DDASDecoder;
class FitManager;
class QHitData;
class QRootCanvas;

/**
 * @class QTraceView
 * @brief Main traceview GUI window.
 *
 * @details
 * Main window class for traceview responsible for management and high-level
 * control over the application. Uses Qt's signal and slot framework to
 * communicate between objects. See Qt documentation for details.
 */

/** 
 * @todo (ASC 10/15/24): Probably a project-wide TODO... only the public and 
 * Qt signaling interfaces need to be exposed here. I suspect that any helper 
 * functions can be make static, which may greatly simplify the class and make 
 * compilation easier.
 */

/** 
 * @todo (ASC 10/16/24): Probably _another_ project-wide TODO... avoid 
 * constructor initializations which use the new keyword. Can use smart ptrs 
 * (preferred?) or simply initialize in the constructor body with the 
 * initialization list for all pointer membvers as nullptr. Hopefully improves 
 * safety e.g., an exception or other error in the constructor body will 
 * (hopefully!) ensure deletion of created objetcs which _may_ not happen if 
 * new'd in init list.
 */

class QTraceView : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Constructor.
     * @param parser References the QCommandLineParser.
     * @param parent Pointer to QWidget parent object (optional, 
     *   default=Q_NULLPTR).
     */
    QTraceView(QCommandLineParser& parser, QWidget* parent=Q_NULLPTR);
    /** @brief Destructor. */
    virtual ~QTraceView();

    // QEvent handlers overridden from the base class.  
protected:
    /**
     * @brief Event handler for state changes. 
     * @param e Pointer to the handled QEvent.
     */
    virtual void changeEvent(QEvent* e);

private slots:
    /** @brief Open an NSCLDAQ event file. */
    void openFile();
    /** @brief Get the next event. */
    void getNextEvent();
    /** 
     * @brief Get the next event containing trace data matching the 
     * selection criteria. 
     */
    void getNextEventWithTraces();
    /** @brief Get the next event from the event list provided at runtime. */
    void getNextEventFromList();
    /** @brief Skip events in the source. */
    void skipEvents();
    /** @brief Select a specific event. */
    void selectEvent();
    /** @brief Update the hit selection list. */
    void updateSelectableHits();
    /** @brief Process a selected hit from the hit selection list. */
    void processHit();
    /** @brief Apply the hit filter to the hits for the current event. */
    void filterHits();
    /** @brief Process pending ROOT events. */
    void handleRootEvents();
    /** @brief Test slot function which prints the current time to stdout. */
    void test();
    
private:
    /** @brief Create the top menu bar. */
    void createMenu();
    /** @brief Create the top group box widgets. */
    void createTopBoxWidget();
    /** @brief Create and configure the hit selection list widget. */
    void createHitSelectWidget();
    /** @brief Create the plotting widget. */
    void createPlotWidget();

    /** 
     * @brief Attempt to create the file data soruce. Update GUI if successful.
     * @param filename Filename as a QString, without URI formatting.
     */
    void configureSource(QString filename);    
    /** 
     * @brief Create signal/slot connections for the main window. 
     * @param useEvtList True if an event list is provided (optional, 
     *   default=false).
     */
    void createConnections(bool useEvtList=false);

    /**
     * @brief Skip ahead and select the next event.
     * @param count How far to skip ahead.
     */
    void skipAndSelect(int count);
    /**
     * @brief Select an event by its index.
     * @param idx Which PHYISCS_EVENT to view.
     */
    void selectEventByIndex(int idx);
    /**   
     * @brief Check if the current hit passes the hit filter. 
     * @param hit References the hit to validate.
     * @return True if the hit passes the filter, false otherwise.
     */
    bool isValidHit(const ddastoys::DDASFitHit& hit);    
    
    /**
     * @brief Set the status bar message.
     * @param msg Message displayed on the status bar.
     */
    void setStatusBar(std::string msg);    
    /**
     * @brief Reset and clear all GUI elements and member data to default 
     * states.
     */
    void reset();
    /** 
     * @brief Set the UI button status. 
     * @param status True to enable buttons, false to disable.
     */
    void enableGUI(bool status);
    /** 
     * @brief Set the main button status. 
     * @param status True to enable buttons, false to disable.
     */
    void enableMainGUI(bool status);
    /** 
     * @brief Set the event selection button status. 
     * @param status True to enable buttons, false to disable.
     */
    void enableSelectGUI(bool status);
	
    /**   
     * @brief Issue a warning message in a popup window.
     * @param msg The warning message displayed in the popup window.
     */
    void issueWarning(std::string msg);
    /** @brief Issue an EOF warning when we're out of PHYSICS_EVENT data. */
    void issueEOFWarning();
		      
    /**
     * @brief Parse command line arguments supplied at runtime.
     * @param parser References the QCommandLineParser of the main 
     *   QApplication.
     */
    void parseArgs(QCommandLineParser& parser);
    /** 
     * @brief Read the event list into memory. 
     * @param fname File containing the events to plot, one per line.
     * @throw std::runtime_error If the event list file cannot be opened.
     */
    void parseEventList(QString fname);
    
private:  
    // Our member data  
    DDASDecoder* m_pDecoder;   //!< Decoder to perform event processing.
    FitManager*  m_pFitManager; //!< Manager for calculating fits from params.

    std::vector<ddastoys::DDASFitHit> m_hits; //!< List of hits in the event.
    std::vector<ddastoys::DDASFitHit> m_filteredHits; //!< Hits passed filter.
    /** 
     * @brief A struct for storing a user-provided subset of events and an 
     * index tracking which event is currently being viewed.
     */
    struct EventList {
	std::vector<int> s_events; //!< List of event numbers.
	unsigned long s_idx; //!< Current event e.g., for loop over list.
	EventList() {};
	EventList(std::vector<int> evts, unsigned long idx=0)
	    : s_events(evts), s_idx(idx) {}
    };
    EventList m_eventList; //!< Pre-selected events to view.
  
    // Added to this widget, Qt _should_ handle cleanup on destruction
  
    QMenuBar* m_pMenuBar;   //!< Top menu bar.
    QMenu*    m_pFileMenu;  //!< File menu on the top menu bar.
    QAction*  m_pOpenAction; //!< Open a file and crate a data source from menu.
    QAction*  m_pExitAction; //!< Clean exit the program.

    QPushButton* m_pMainButtons[3]; //!< "Next", "Update" and "Exit" buttons.
    QPushButton* m_pSelectButtons[2]; //!< "Skip" and "Select" buttons.
    QLineEdit*   m_pSelectLineEdit; //!< Events to skip or select.
    QLineEdit*   m_pHitFilter[3]; //!< Crate/slot/channel filter values.
    QWidget*     m_pTopBoxes; //!< Layout for selection and event handling.
    QHitData*    m_pHitData; //!< Display widget for hit data.
  
    QListView*   m_pHitSelectList; //!< The list to select channel hits to draw.
    QWidget*     m_pPlotWidget; //!< Widget containing the plotting canvas.
    QRootCanvas* m_pRootCanvas; //!< ROOT canvas to display hits.
    QTimer*      m_pTimer; //!< Timer used for ROOT event handling.
  
    QStatusBar*  m_pStatusBar; //!< Shows status messages.
};

#endif
