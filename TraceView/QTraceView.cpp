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
 * @file  QTraceView.cpp
 * @brief Implement Qt main application window class.
 */

#include "QTraceView.h"

#include <iostream>
#include <ctime>

#include <QVBoxLayout>
#include <QStatusBar>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QComboBox>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QListView>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QTimer>
#include <QEvent>
#include <QWindowStateChangeEvent>
#include <QCommandLineParser>
#include <QMessageBox>

#include <TSystem.h>
#include <TCanvas.h>

#include <DDASFitHit.h>
#include "FitManager.h"
#include "DDASDecoder.h"
#include "QHitData.h"
#include "QRootCanvas.h"

using namespace ddastoys;

//____________________________________________________________________________
/**
 * @details
 * Constructs the widgets which define the main window layout. Initialization 
 * list order and the order of the funciton calls in the constructor matter as 
 * some member variables depend on others being initialized to configure 
 * children or actions.
 */
QTraceView::QTraceView(QCommandLineParser& parser, QWidget* parent) :
    QWidget(parent), m_pDecoder(new DDASDecoder), m_pFitManager(new FitManager),
    m_pMenuBar(new QMenuBar), m_pFileMenu(nullptr), m_pOpenAction(nullptr),
    m_pExitAction(nullptr), m_pMainButtons{nullptr}, m_pSelectButtons{nullptr},
    m_pSelectLineEdit(nullptr), m_pHitFilter{nullptr},
    m_pTopBoxes(createTopBoxes()), m_pHitData(new QHitData(m_pFitManager)),
    m_pHitSelectList(createHitSelectList()),
    m_pRootCanvas(new QRootCanvas(m_pFitManager)), m_pTimer(new QTimer),
    m_pStatusBar(new QStatusBar)
{ 
    createActions();
    configureMenu();
  
    // Combine hit selection list and canvas into a single horizontally
    // aligned widget which we add to the main layout:
    
    QWidget* plotWidget = createPlotWidget();  
  
    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->setMenuBar(m_pMenuBar);
    mainLayout->addWidget(m_pTopBoxes);
    mainLayout->addWidget(m_pHitData);
    mainLayout->addWidget(plotWidget);
    mainLayout->addWidget(m_pStatusBar);
    setLayout(mainLayout);

    createConnections();
    disableAll();

    parseArgs(parser);
  
    m_pTimer->start(20);
}

//____________________________________________________________________________
/**
 * @details
 * Qt _should_ manage deletion of child objects when each parent is destroyed
 * upon application exit. But we still should clean up our own stuff.
 */
QTraceView::~QTraceView()
{
    delete m_pDecoder;
    delete m_pFitManager;
}

///
// Private member functions inherited from QWidget
// 

//____________________________________________________________________________
/**
 * @details
 * Primary purpose here is to propagate the state changes to ROOT.
 */
void
QTraceView::changeEvent(QEvent* e)
{
    if (e->type() == QEvent ::WindowStateChange) {
	QWindowStateChangeEvent* event
	    = static_cast<QWindowStateChangeEvent*>(e);
	if ((event->oldState() & Qt::WindowMaximized) ||
	    (event->oldState() & Qt::WindowMinimized) ||
	    (event->oldState() == Qt::WindowNoState && 
	     this->windowState() == Qt::WindowMaximized)) {
	    if (m_pRootCanvas->getCanvas()) {
		m_pRootCanvas->getCanvas()->Resize();
		m_pRootCanvas->getCanvas()->Update();
	    }
	}
    }
}

///
// Private member functions
//

//____________________________________________________________________________
// Create and configure methods

/**
 * @todo (ASC 2/3/23): Creation methods should not fail without explanation -
 * there is an order in which (some of) these functions must be called but no
 * safeguards to prevent someone from doing it wrong e.g. attempting to
 * configure the menu bar before its created.
 */

//____________________________________________________________________________
/** 
 * @details
 * See Qt documentation for more information. Actions for interactions with 
 * the top menu bar should be created here before adding them to their 
 * associated QMenu objects.
 */
void
QTraceView::createActions()
{
    m_pOpenAction = new QAction(tr("&Open File..."), this);
    m_pOpenAction->setStatusTip(tr("Open an existing file"));
    connect(m_pOpenAction, SIGNAL(triggered()), this, SLOT(openFile()));

    m_pExitAction = new QAction(tr("E&xit"), this);
    m_pExitAction->setStatusTip(tr("Exit the application"));
    connect(m_pExitAction, &QAction::triggered, this, &QWidget::close);
}

//____________________________________________________________________________
/**
 * @detials
 * The menu bar is a collection of QMenu objects. Menu actions should be 
 * created prior to their addition.
 */
void
QTraceView::configureMenu()
{
    m_pFileMenu = new QMenu(tr("&File"), this);
    m_pFileMenu->addAction(m_pOpenAction);
    m_pFileMenu->addSeparator();
    m_pFileMenu->addAction(m_pExitAction);

    m_pMenuBar->addMenu(m_pFileMenu);
}

//____________________________________________________________________________
/**
 * @details
 * These widgets contain the channel selection boxes and event handling.
 */
QWidget*
QTraceView::createTopBoxes()
{
    QWidget* w = new QWidget;
    QHBoxLayout* mainLayout = new QHBoxLayout;

    // Channel selection
  
    QGroupBox* channelBox = new QGroupBox;
    channelBox->setTitle("Channel Selection");
    QHBoxLayout* channelBoxLayout = new QHBoxLayout;
    const char* letext[3] = {"Crate:" , "Slot:", "Channel:"};
    for (int i = 0; i < 3; i++) {
	QLabel* label = new QLabel(letext[i]);
	m_pHitFilter[i] = new QLineEdit("*");
	channelBoxLayout->addWidget(label);
	channelBoxLayout->addWidget(m_pHitFilter[i]);
    }
    channelBox->setLayout(channelBoxLayout);

    // Event selection control:
  
    QGroupBox* selectBox = new QGroupBox;
    selectBox->setTitle("Event Selection");
    QHBoxLayout* selectBoxLayout = new QHBoxLayout;
    const char* sbtext[2] = {"Skip", "Select"};
    for (int i = 0; i < 2; i++) {
	m_pSelectButtons[i] = new QPushButton(tr(sbtext[i]));
	selectBoxLayout->addWidget(m_pSelectButtons[i]);	
    }
    m_pSelectLineEdit = new QLineEdit("0");
    m_pSelectLineEdit->setValidator(new QIntValidator(0, 9999999, this));
    selectBoxLayout->addWidget(m_pSelectLineEdit);
    selectBox->setLayout(selectBoxLayout);

    // Main control (next event, update, exit):
  
    QGroupBox* mainBox = new QGroupBox;
    mainBox->setTitle("Main Control");
    QHBoxLayout* mainBoxLayout = new QHBoxLayout;
    const char* mbtext[3] = {"Next", "Update", "Exit"};
    for (int i = 0; i < 3; i++) {
	m_pMainButtons[i] = new QPushButton(tr(mbtext[i]));
	mainBoxLayout->addWidget(m_pMainButtons[i]);
    }
    mainBox->setLayout(mainBoxLayout);

    // Setup the actual widget:
  
    mainLayout->addWidget(channelBox);
    mainLayout->addWidget(selectBox);
    mainLayout->addWidget(mainBox);
    w->setLayout(mainLayout);
  
    return w;
}

//____________________________________________________________________________
QListView*
QTraceView::createHitSelectList()
{    
    QListView* listView = new QListView;
    QStandardItemModel* model = new QStandardItemModel;
    listView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    listView->setSelectionMode(QAbstractItemView::SingleSelection);
    listView->setModel(model);

    return listView;
}

//____________________________________________________________________________
/**
 * @brief Create signal/slot connections for the main window. 
 *
 * @details
 * See Qt signal/slot documentation for more information.
 */
void
QTraceView::createConnections()
{
    // Skip button
    connect(m_pSelectButtons[0], SIGNAL(clicked()), this, SLOT(skipEvents()));
    
    // Select button
    connect(m_pSelectButtons[1], SIGNAL(clicked()), this, SLOT(selectEvent()));
  
    // Next button  
    connect(
	m_pMainButtons[0], SIGNAL(clicked()),
	this, SLOT(getNextEventWithTraces())
	);

    // Update buttons  
    connect(m_pMainButtons[1], SIGNAL(clicked()), this, SLOT(filterHits()));
    connect(
	m_pMainButtons[1], SIGNAL(clicked()),
	this, SLOT(updateSelectableHits())
	);
  
    // Exit button  
    connect(m_pMainButtons[2], SIGNAL(clicked()), this, SLOT(close()));

    // Hit selection  
    connect(
	m_pHitSelectList->selectionModel(),
	SIGNAL(selectionChanged(QItemSelection, QItemSelection)),
	this, SLOT(processHit())
	);

    // Timer to call inner loop of ROOT  
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(handleRootEvents()));  
}

//____________________________________________________________________________
/**
 * @details 
 * Combine the hit selection list and the ROOT canvas into a single widget 
 * with its own layout which can be added to the main window layout.
 */
QWidget*
QTraceView::createPlotWidget()
{
    // Set a label for the selection list  
    QWidget* labeledList = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout;
    QLabel* label = new QLabel("Crate:Slot:Channel Hits with Traces");
    layout->addWidget(label);
    layout->addWidget(m_pHitSelectList);
    labeledList->setLayout(layout);
  
    // Combine channel list and plot into a widget  
    QWidget* plot = new QWidget;
    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addWidget(labeledList);
    mainLayout->addWidget(m_pRootCanvas);
    plot->setLayout(mainLayout);

    return plot;
}

//____________________________________________________________________________
// Utilities

//____________________________________________________________________________
void
QTraceView::setStatusBar(std::string msg)
{
    m_pStatusBar->showMessage(tr(msg.c_str()));
}

//____________________________________________________________________________
/**
 * @details
 * Valid events match the crate/slot/channel values set in the filter boxes. 
 * Wildcard '*' characters pass everything. Valid hits must contain traces.
 */
bool
QTraceView::isValidHit(const DDASFitHit& hit)
{
    bool crateMatch = false;
    bool slotMatch = false;
    bool channelMatch = false;
    bool hasTrace = false;
    if ((m_pHitFilter[0]->text() == "*") ||
	(m_pHitFilter[0]->text().toUInt() == hit.getCrateID())) {
	crateMatch = true;
    }
    if ((m_pHitFilter[1]->text() == "*") ||
	(m_pHitFilter[1]->text().toUInt() == hit.getSlotID())) {
	slotMatch = true;
    }
    if ((m_pHitFilter[2]->text() == "*") ||
	(m_pHitFilter[2]->text().toUInt() == hit.getChannelID())) {
	channelMatch = true;
    }

    std::vector<uint16_t> trace = hit.getTrace();
    if (!trace.empty()) {
	hasTrace = true;
    }
  
    return (crateMatch && slotMatch && channelMatch && hasTrace);
}

//____________________________________________________________________________
/**
 * @details
 * The list is updated based on the current list of filtered hits. Each hit is 
 * a formatted QString crate:slot:channel where the idenfiying information is 
 * read from the hit itself. The QStandardItemModel is used to provide data to 
 * the QListView interface, see Qt documentation for details.
 */
void
QTraceView::updateSelectableHits()
{
    QStandardItemModel* model
	= reinterpret_cast<QStandardItemModel*>(m_pHitSelectList->model());
    model->clear();

    for (unsigned i = 0; i < m_filteredHits.size(); i++) {    
	// Qt 5.14+ supports arg(arg1, arg2, ...) but we're stuck with this    
	QString id = QString("%1:%2:%3").arg(m_filteredHits[i].getCrateID()).arg(m_filteredHits[i].getSlotID()).arg(m_filteredHits[i].getChannelID());    
	QStandardItem* item = new QStandardItem(id);
	model->setItem(i, item);
    }
}

//____________________________________________________________________________
void
QTraceView::resetGUI()
{
    m_hits.clear();
    m_filteredHits.clear();
    updateSelectableHits();
    m_pRootCanvas->clear();
    m_pStatusBar->showMessage("");
}

//____________________________________________________________________________
void
QTraceView::enableAll()
{
    for (int i = 0; i < 3; i++) {
	m_pMainButtons[i]->setEnabled(true);
    }
    for (int i = 0; i < 2; i++) {
	m_pSelectButtons[i]->setEnabled(true);
    }
    m_pSelectLineEdit->setEnabled(true);
}

//____________________________________________________________________________
/** 
 * @details
 * Disable everything except exit, which we should always be able to do. Assume
 * the exit button is the last one in the button list.
 */
void
QTraceView::disableAll()
{
    for (int i = 0; i < 2; i++) {
	m_pMainButtons[i]->setEnabled(false);
    }
    for (int i = 0; i < 2; i++) {
	m_pSelectButtons[i]->setEnabled(false);
    }
    m_pSelectLineEdit->setEnabled(false);
}

//____________________________________________________________________________
/**
 * @details
 * Use QCommandLineParser to read in and parse positional arguments supplied 
 * on the command line when the program is run. All arguments are considered 
 * optional. Handle the arguments which are provided and issue error messages 
 * if necessary.
 */
void
QTraceView::parseArgs(QCommandLineParser& parser)
{
    const QStringList args = parser.positionalArguments();
    const QStringList names = parser.optionNames();

    bool sourceSet = parser.isSet("source");
    if (sourceSet) {
	QString filename = parser.value("source");
	configureSource(filename);
    }

    bool methodSet = parser.isSet("method");
    if (methodSet) {
	QString method = parser.value("method");
	try {
	    m_pHitData->setFitMethod(method);
	} catch (std::invalid_argument& e) {
	    std::cerr << "QTraceView::parseArgs(): Unknown fitting method "
		      << method.toStdString() << " read from command line."
		      << " Setting fit method to 'Analytic'." << std::endl;
	    m_pHitData->setFitMethod("Analytic");
	}
    }  
}

///
// Private slots
//

//____________________________________________________________________________
/**
 * @details
 * Open a file using the QFileDialog and attempt to create a data source 
 * from it. Update status bar to show the currently loaded file and enable 
 * UI elements on the main window.
 */
void
QTraceView::openFile()
{
    QString filename = QFileDialog::getOpenFileName(
	this, tr("Open File"), "", tr("Event files (*.evt);;All Files (*)")
	);

    // Get the current file path from the decoder. The path is an empty string
    // if no data source has been configured.  
    std::string currentPath = m_pDecoder->getFilePath();
  
    if (filename.isEmpty() && currentPath.empty()) {
	std::cout << "WARNING: no file selected."
		  << " Please open a file using File->Open file..."
		  << " before continuing." << std::endl;
	return;
    } else if (filename.isEmpty() && !currentPath.empty()) {
	std::cout << "WARNING: no file selected,"
		  << " but a data source already exists."
		  << " Current file path is: " << currentPath << std::endl;
    } else {
	configureSource(filename);
    }
}

//____________________________________________________________________________
/**
 * @details
 * Create a data source from a filename string. Update status bar to show the
 * currently loaded file and enable UI elements on the main window. Errors 
 * which occur during data source creation are dealt with in the decoder.
 */
void
QTraceView::configureSource(QString filename)
{
    m_pDecoder->createDataSource(filename.toStdString());
    resetGUI();
    setStatusBar(filename.toStdString());
    enableAll();
}

//____________________________________________________________________________
/**
 * @details
 * Get the next PHYSICS_EVENT event from the data source using the DDASDecoder.
 * Apply the event filter and update the UI.
 * @note Depending on the channel selection filter settings and whether the hit
 * has trace data, the hit list may be empty.
 */
void
QTraceView::getNextEvent()
{
    m_hits = m_pDecoder->getEvent();

    // If the hit list is empty there are no more PHYSICS_EVENTs to grab
    // so issue the EOF warning and return from the function.    
    filterHits();
  
    if (m_hits.empty()) {
	issueEOFWarning();		
	return;
    }

    updateSelectableHits();
    m_pRootCanvas->clear();
  
    std::string msg = m_pDecoder->getFilePath()
	+ " -- Event " + std::to_string(m_pDecoder->getEventIndex());
    setStatusBar(msg);
}

//____________________________________________________________________________
/**
 * @details
 * Clear the list of filtered hits and keep grabbing events until the filtered 
 * hit list is not empty i.e., there is at least one valid hit according to 
 * the channel selection criteria.
 */
void
QTraceView::getNextEventWithTraces()
{    
    m_filteredHits.clear();  
    while (m_filteredHits.empty()) {
	getNextEvent();
    }
}

//____________________________________________________________________________
/** 
 * @details
 * The number of events to skip is read from the QLineEdit box when the skip 
 * button is clicked. If the end of the source file is encountered while 
 * skipping ahead, pop up a notification to that effect.
 */
void
QTraceView::skipEvents()
{
    setStatusBar("Skipping events, be patient...");
  
    int skipCount = m_pSelectLineEdit->text().toInt();
    int retval = m_pDecoder->skip(skipCount);

    if (retval < 0) {   
	issueEOFWarning();
	return;
    } else {
	getNextEvent();
    }
}

/**
 * @details
 * 
 */ 
void
QTraceView::selectEvent()
{
    int select = m_pSelectLineEdit->text().toInt();
    int current = m_pDecoder->getEventIndex();
    int skipCount = select - current;

    // We're currently looking at this event:

    if (select == current) {
	return;
    }

    // We've advanced in the file and are trying to go backwards.
    // Issue a warning and don't do anything:
    
    if (current >= 0 && skipCount < 0) {
	issueWarning(
	    "Current event = " + std::to_string(current)
	    + ", selected event = " + std::to_string(select)
	    + "\nCannot select previous events!\n"
	    );
	return;
    }

    // Otherwise skip and select:
    
    int retval = m_pDecoder->skip(skipCount-1); // We want the next hit...    
    if (retval < 0) {   
	issueEOFWarning();
	return;
    } else {
	getNextEvent(); // ...which we grab here if we're not at EOF.
    }
}

//____________________________________________________________________________
/**
 * @brief Apply the hit filter to the current list of hits for this event.
 */
void
QTraceView::filterHits()
{
    if (!m_filteredHits.empty()) {
	m_filteredHits.clear();
    }
    for (auto& hit : m_hits) {
	if (isValidHit(hit)) {
	    m_filteredHits.push_back(hit);
	}
    }
}

//____________________________________________________________________________
/**
 * @details
 * The row number in the hit select list corresponds to the index of that hit 
 * in the list of filtered hits. Draw the trace and hit information and update
 * the hit data display.
 */
void
QTraceView::processHit()
{
    QItemSelectionModel* itemSelect = m_pHitSelectList->selectionModel();
    int idx = itemSelect->currentIndex().row();
    m_pRootCanvas->drawHit(m_filteredHits[idx]);
    m_pHitData->update(m_filteredHits[idx]);  
}

//____________________________________________________________________________
void
QTraceView::handleRootEvents()
{
    gSystem->ProcessEvents();
}

//____________________________________________________________________________
/** 
 * @details
 * The warning is issued as a modal dialog, blocking until the user closes it.
 */
void
QTraceView::issueWarning(std::string msg)
{
    QMessageBox msgBox;
    msgBox.setText(QString::fromStdString(msg));
    msgBox.setIcon(QMessageBox::Warning);
    msgBox.exec();
}

/** 
 * @details
 * Create the EOF message and issue the warning.
 */
void
QTraceView::issueEOFWarning()
{
    m_hits.clear();
    m_filteredHits.clear();
    updateSelectableHits();
    m_pRootCanvas->clear();
    
    std::string msg("No more physics events in this file.");
    msg += "The file contains ";
    msg += std::to_string(m_pDecoder->getEventCount());
    msg += " physics events.";
    
    issueWarning(msg);
}

//____________________________________________________________________________
/**
 * @brief Test slot function which prints the current time to stdout. 
 */
void
QTraceView::test()
{
    std::time_t result = std::time(nullptr);
    std::cout << "Test slot call at: "
	      << std::asctime(std::localtime(&result))
	      << std::endl;
}
