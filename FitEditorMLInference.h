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
 * @file  FitEditorMLInference.h
 * @brief Definition of the FitEditor class for machine-learning inference.
 */

#ifndef FITEDITORMLINFERENCE_H
#define FITEDITORMLINFERENCE_H

#include <CBuiltRingItemEditor.h>

#include <vector>
#include <string>
#include <map>

#include <torch/script.h> // Can forward-declare?
#include <torch/torch.h>  // Ditto?


/** @namespace ddastoys */
namespace ddastoys {
    
    class Configuration;

    /**
     * @class FitEditorMLInference
     * @brief Fit trace data using machine-learning inference with a detector 
     * response model defined by the analytic fit functions.
     *
     * @details
     * Extending the hit with this editor overwrites any existing extension. 
     * This class is intended for use with the EventEditor framework providing 
     * a complete description of the new event body.
     */
 
    class FitEditorMLInference : public CBuiltRingItemEditor::BodyEditor
    {
    public:
	/** @brief Constructor. */
	FitEditorMLInference();
	/**
	 * @brief Copy constructor.
	 * @param rhs Object to copy construct.
	 */
	FitEditorMLInference(const FitEditorMLInference& rhs);    
	/**
	 * @brief Move constructor.
	 * @param rhs Object to move construct.
	 */
	FitEditorMLInference(FitEditorMLInference&& rhs) noexcept;

	/**
	 * @brief Copy assignment operator.
	 * @param rhs Object to copy assign.
	 * @return Reference to created object.
	 */
	FitEditorMLInference& operator=(const FitEditorMLInference& rhs);
	/**
	 * @brief Move assignment operator.
	 * @param rhs Object to move assign.
	 * @return Reference to created object.
	 */
	FitEditorMLInference& operator=(FitEditorMLInference&& rhs) noexcept;
    
	/** @brief Destructor. */
	virtual ~FitEditorMLInference();

	// Mandatory interface from CBuiltRingItemEditor::BodyEditor
    public:
	/**
	 * @brief Perform the fit and create a fit extension for a single 
	 * fragment. 
	 * @param pHdr     Pointer to the ring item header of the hit.
	 * @param pBHdr    Pointer to the body header pointer for the hit.
	 * @param bodySize Number of bytes in the body.
	 * @param pBody    Pointer to the body.
	 * @return         Final segment descriptors.
	 */
	virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(
	    pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize,
	    void* pBody
	    );
	/**
	 * @brief Free the dynamic fit extension descriptor(s).
	 * @param e IOvec we need to free.
	 */
	virtual void free(iovec& e);
  
	// Private member data
    private:
	Configuration* m_pConfig; //!< Configuration file parser.
	/** Unique models keyed by path to PyTorch file. */
	std::map<std::string, torch::jit::script::Module> m_models;    
    };

}

#endif
