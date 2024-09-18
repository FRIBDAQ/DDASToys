/** 
 * @file  FitEditorTemplate.h
 * @brief Definition of the FitEditor class for template fitting.
 */

#ifndef FITEDITORTEMPLATE_H
#define FITEDITORTEMPLATE_H

#include <CBuiltRingItemEditor.h>

#include <vector>

namespace DAQ {
    namespace DDAS {
	class DDASHit;
    }
}

namespace ddastoys {
    
    class Configuration;

/**
 * @ingroup template
 * @{
 */

/**
 * @class FitEditorTemplate
 * @brief Fit trace data with the template fitting functions and add hit 
 * extensions.
 *
 * @details
 * Editing the hit overwrites any existing extension. This class is intended 
 * for use with the EventEditor framework providing a complete description of 
 * the new event body.
 */

    class FitEditorTemplate : public CBuiltRingItemEditor::BodyEditor
    {
    public:
	/** @brief Constructor. */
	FitEditorTemplate();
	/**
	 * @brief Copy constructor.
	 * @param rhs Object to copy construct.
	 */
	FitEditorTemplate(const FitEditorTemplate& rhs);
	/**
	 * @brief Move constructor.
	 * @param rhs Object to move construct.
	 */
	FitEditorTemplate(FitEditorTemplate&& rhs) noexcept;

	/**
	 * @brief Copy assignment operator.
	 * @param rhs Object to copy assign.
	 * @return Reference to created object.
	 */
	FitEditorTemplate& operator=(const FitEditorTemplate& rhs);
	/**
	 * @brief Move assignment operator.
	 * @param rhs Object to move assign.
	 * @return Reference to created object.
	 */
	FitEditorTemplate& operator=(FitEditorTemplate&& rhs) noexcept;
    
	/** @brief Destructor. */
	virtual ~FitEditorTemplate();

	// Mandatory interface from CBuiltRingItemEditor::BodyEditor
    public:
	/**
	 * @brief Perform the fit and create a fit extension for a single fragment. 
	 * @param pHdr     Pointer to the ring item header of the hit.
	 * @param pBHdr    Pointer to the body header pointer for the hit.
	 * @param bodySize Number of bytes in the body.
	 * @param pBody    Pointer to the body.
	 * @return Final segment descriptors.
	 */
	virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(
	    pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody
	    );
	/**
	 * @brief Free the dynamic fit extension descriptor(s).
	 * @param e  IOvec we need to free.
	 */
	virtual void free(iovec& e);

	// Additional functionality for this class
    private:
	/**
	 * @brief This is a hook into which to add the ML classifier.
	 * @param hit - references a hit.
	 * @return int
	 * @retval 0  - On the basis of the trace no fitting.
	 * @retval 1  - Only fit a single trace.
	 * @retval 2  - Only fit two traces.
	 * @retval 3  - Fit both one and double hit.
	 */
	int pulseCount(DAQ::DDAS::DDASHit& hit);
  
	// Private member data
    private:
	Configuration* m_pConfig; //! Configuration file parser.
	std::vector<double> m_template; //!< Trace template.
	unsigned m_align; //!< Alignment point on the trace.
    };

/** @} */

}

#endif
