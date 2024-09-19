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
 * @file  QHitData.h
 * @brief Definition of class to handle disaplying hit data on the main 
 * window UI.
 */

#ifndef QHITDATA_H
#define QHITDATA_H

#include <QWidget>

#include <string>

class QLabel;
class QComboBox;
class QPushButton;
class QGroupBox;
class QString;

namespace ddastoys {
    struct HitExtension;
    class DDASFitHit;
}

class FitManager;

/**
 * @class QHitData
 * @brief Widget for managing hit information when interating with the GUI.
 *
 * @details
 * Widget class for managing UI related to the hit data. Displays relavent 
 * infomration for the currently displayed hit with optional UI elements to
 * see more information e.g. a button to print the fit results to stdout.
 */

class QHitData : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Constructor.
     * @param pFitMgr Pointer to FitManager object used by this class, managed 
     *   by caller.
     * @param parent Pointer to QWidget parent object (optional).
     */
    QHitData(FitManager* pFitMgr, QWidget* parent=nullptr);
    /** @brief Destructor. */
    ~QHitData();

    /**
     * @brief Update hit data and enable printing of fit information to stdout 
     * if the hit has an extension.
     * @param hit References the hit we are processing
     */ 
    void update(const ddastoys::DDASFitHit& hit);
    /**
     * @brief Set the fit method.
     * @param method  The name of the fitting method.
     * @throw std::invalid_argument If the method parameter does not match a 
     *   known fitting method.
     */
    void setFitMethod(QString method);

private:
    /**
     * @brief Create and configure the hit group box containing widgets to 
     * display basic hit information.
     * @return Pointer to the created QGroupBox object.
     */
    QGroupBox* createHitBox();
    /**
     * @brief Create and configure the classifier group box containing widgets
     * to display machine learning pulse classifier probabilities.
     * @return Pointer to the created QGroupBox object.
     */
    QGroupBox* createClassifierBox();
    /**
     * @brief Create and configure the fit group box containing widgets to 
     * select a fit method and print fit results to stdout.
     * @return Pointer to the created QGroupBox object
     */
    QGroupBox* createFitBox();
    /**
     * @brief Create signal/slot connections for the hit data top widget. 
     * See Qt documentation for more information.
     */
    void createConnections();
    /**
     * @brief Update the data displayed in the hit data group box. Basic hit 
     * information contains at minimum and ID (crate/slot/channel), an energy
     * and a time.
     * @param hit References the hit we are displaying data for.
     */
    void updateHitData(const ddastoys::DDASFitHit& hit);

private slots:
    /** @brief Read the fitting method and configure the FitManager. */
    void configureFit();
    /**
     * @brief Print formatted fit results for the single and double pulse fits 
     * to stdout.
     */
    void printFitResults();
  
private:
    FitManager* m_pFitManager;        //!< Manager for fits, owned by caller.
    ddastoys::HitExtension* m_pExtension; //!< For the currently selected hit.

    // Data for currently selected hit.
    
    QLabel* m_pId;       //!< Global ID value.
    QLabel* m_pRawData;  //!< Energy and timestamp.
    QLabel* m_pFit1Prob; //!< Single-pulse prob. from ML classifier
    //!< (not implemented) 
    QLabel* m_pFit2Prob; //!< Double-pulse prob. from ML classifier
    //!< (not implemented) 
    QComboBox* m_pFitMethod;  //!< Fit method selection box.
    QPushButton* m_pPrintFit; //!< Print button.
};

#endif
