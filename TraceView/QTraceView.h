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

    class QTraceView : public QWidget
    {
	Q_OBJECT

    public:
	/**
	 * @brief Constructor.
	 * @param parser References the QCommandLineParser. The parser is owned by 
	 *   the caller (in this case the application main() function).
	 * @param parent (optional) Pointer to QWidget parent object.
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

    private:
	/**  @brief Create commands (user actions). */
	void createActions();
	/** @brief Create and configure the top menu bar. */
	void configureMenu();
	/** 
	 * @brief Create the top group box widgets. 
	 * @return Pointer to the created QGroupBox object.
	 */
	QWidget* createTopBoxes();
	/**
	 * @brief Create and configure the hit selection list widget.
	 * @return Pointer to the created QListView widget.
	 */
	QListView* createHitSelectList();
	/** @brief Create signal/slot connections for the main window. */
	void createConnections();
	/**  
	 * @brief Create the plotting widget.
	 * @return  Pointer to the created QWidget object.
	 */
	QWidget* createPlotWidget();
	/**
	 * @brief Set the status bar message.
	 * @param msg  Message displayed on the status bar.
	 */
	void setStatusBar(std::string msg);
	/**   
	 * @brief Check if the current hit passes the hit filter. 
	 * @param hit References the hit to validate.
	 * @return True if the hit passes the filter, false otherwise.
	 */
	bool isValidHit(const ddastoys::DDASFitHit& hit);
	/**
	 * @brief Reset and clear all GUI elements and member data to default 
	 * states.
	 */
	void resetGUI();
	/** @brief Enable all UI buttons. */
	void enableAll();
	/** @brief Disable all UI buttons. */
	void disableAll();
	/**
	 * @brief Parse command line arguments supplied at runtime.
	 * @param parser References the QCommandLineParser of the main 
	 *   QApplication.
	 */
	void parseArgs(QCommandLineParser& parser);
			   
    private slots:
	/** @brief Open an NSCLDAQ event file. */
	void openFile();
	/** 
	 * @brief Attempt to create the file data soruce. Update GUI if successful.
	 * @param filename Filename as a QString, without URI formatting.

	 */
	void configureSource(QString filename);
	/** @brief Get the next event containing trace data. */
	void getNextEvent();
	/** @brief Skip events in the source. */
	void skipEvents();
	/** @brief Apply the hit filter to the current hits for this event. */
	void filterHits();
	/** @brief Update the hit selection list. */
	void updateSelectableHits();
	/** @brief Process a selected hit from the hit selection list. */
	void processHit();
	/** @brief Process pending Root events. */
	void handleRootEvents();
	/**   
	 * @brief Issue a warning message in a popup window.
	 * @param msg  The warning message displayed in the popup window.
	 */
	void issueWarning(std::string msg);
	/** @brief Test slot function which prints the current time to stdout. */
	void test();
  
    private:  
	// Our member data  
	DDASDecoder* m_pDecoder;   //!< Decoder to perform event processing.
	FitManager* m_pFitManager; //!< Manager for calculating fits from params.

	std::vector<DDASFitHit> m_hits; //!< List of hits in the event.
	std::vector<DDASFitHit> m_filteredHits; //!< Hits passed filter.
  
	// Added to this widget, Qt _should_ handle cleanup on destruction
  
	QMenuBar* m_pMenuBar;   //!< Top menu bar.
	QMenu* m_pFileMenu;     //!< File menu on the top menu bar.
	QAction* m_pOpenAction; //!< Open a file and crate a data source from menu.
	QAction* m_pExitAction; //!< Clean exit the program.

	QPushButton* m_pButtons[3]; //!< The "Next", "Update" and "Exit" buttons.
	QPushButton* m_pSkipEvents; //!< Skip events button.
	QLineEdit* m_pEventsToSkip; //!< Contains number of events to skip.
	QLineEdit* m_pHitFilter[3]; //!< Crate/slot/channel filter values (wildcard
	//!< "*" is OK).
	QWidget* m_pTopBoxes; //!< Defines layout for selection and event handling
	//!< widget group boxes.
	QHitData* m_pHitData; //!< Display widget for hit data.
  
	QListView* m_pHitSelectList; //!< The list to select channel hits to draw.
	QRootCanvas* m_pRootCanvas; //!< ROOT canvas to display hits.
	QTimer* m_pTimer; //!< Timer used for ROOT event handling.
  
	QStatusBar* m_pStatusBar; //!< Shows status messages.
    };
}

#endif
