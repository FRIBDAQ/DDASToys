// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME DDASRootFitFormat
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "RConfig.h"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include "../DDASFitHit.h"
#include "RootExtensions.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ddastoys {
   namespace ROOTDict {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *ddastoys_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("ddastoys", 0 /*version*/, "DDASRootFitEvent.h", 32,
                     ::ROOT::Internal::DefineBehavior((void*)nullptr,(void*)nullptr),
                     &ddastoys_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_DICT_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_DICT_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *ddastoys_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}

namespace DAQ {
   namespace DDAS {
   namespace ROOTDict {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *DAQcLcLDDAS_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("DAQ::DDAS", 0 /*version*/, "DDASHit.h", 34,
                     ::ROOT::Internal::DefineBehavior((void*)nullptr,(void*)nullptr),
                     &DAQcLcLDDAS_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_DICT_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_DICT_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *DAQcLcLDDAS_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace ROOT {
   static void *new_ddastoyscLcLDDASRootFitEvent(void *p = nullptr);
   static void *newArray_ddastoyscLcLDDASRootFitEvent(Long_t size, void *p);
   static void delete_ddastoyscLcLDDASRootFitEvent(void *p);
   static void deleteArray_ddastoyscLcLDDASRootFitEvent(void *p);
   static void destruct_ddastoyscLcLDDASRootFitEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::DDASRootFitEvent*)
   {
      ::ddastoys::DDASRootFitEvent *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::DDASRootFitEvent >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::DDASRootFitEvent", ::ddastoys::DDASRootFitEvent::Class_Version(), "DDASRootFitEvent.h", 55,
                  typeid(::ddastoys::DDASRootFitEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::DDASRootFitEvent::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::DDASRootFitEvent) );
      instance.SetNew(&new_ddastoyscLcLDDASRootFitEvent);
      instance.SetNewArray(&newArray_ddastoyscLcLDDASRootFitEvent);
      instance.SetDelete(&delete_ddastoyscLcLDDASRootFitEvent);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLDDASRootFitEvent);
      instance.SetDestructor(&destruct_ddastoyscLcLDDASRootFitEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::DDASRootFitEvent*)
   {
      return GenerateInitInstanceLocal((::ddastoys::DDASRootFitEvent*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitEvent*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static TClass *DAQcLcLDDAScLcLDDASHit_Dictionary();
   static void DAQcLcLDDAScLcLDDASHit_TClassManip(TClass*);
   static void *new_DAQcLcLDDAScLcLDDASHit(void *p = nullptr);
   static void *newArray_DAQcLcLDDAScLcLDDASHit(Long_t size, void *p);
   static void delete_DAQcLcLDDAScLcLDDASHit(void *p);
   static void deleteArray_DAQcLcLDDAScLcLDDASHit(void *p);
   static void destruct_DAQcLcLDDAScLcLDDASHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::DAQ::DDAS::DDASHit*)
   {
      ::DAQ::DDAS::DDASHit *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::DAQ::DDAS::DDASHit));
      static ::ROOT::TGenericClassInfo 
         instance("DAQ::DDAS::DDASHit", "DDASHit.h", 78,
                  typeid(::DAQ::DDAS::DDASHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &DAQcLcLDDAScLcLDDASHit_Dictionary, isa_proxy, 4,
                  sizeof(::DAQ::DDAS::DDASHit) );
      instance.SetNew(&new_DAQcLcLDDAScLcLDDASHit);
      instance.SetNewArray(&newArray_DAQcLcLDDAScLcLDDASHit);
      instance.SetDelete(&delete_DAQcLcLDDAScLcLDDASHit);
      instance.SetDeleteArray(&deleteArray_DAQcLcLDDAScLcLDDASHit);
      instance.SetDestructor(&destruct_DAQcLcLDDAScLcLDDASHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::DAQ::DDAS::DDASHit*)
   {
      return GenerateInitInstanceLocal((::DAQ::DDAS::DDASHit*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::DAQ::DDAS::DDASHit*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *DAQcLcLDDAScLcLDDASHit_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::DAQ::DDAS::DDASHit*)nullptr)->GetClass();
      DAQcLcLDDAScLcLDDASHit_TClassManip(theClass);
   return theClass;
   }

   static void DAQcLcLDDAScLcLDDASHit_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *ddastoyscLcLPulseDescription_Dictionary();
   static void ddastoyscLcLPulseDescription_TClassManip(TClass*);
   static void *new_ddastoyscLcLPulseDescription(void *p = nullptr);
   static void *newArray_ddastoyscLcLPulseDescription(Long_t size, void *p);
   static void delete_ddastoyscLcLPulseDescription(void *p);
   static void deleteArray_ddastoyscLcLPulseDescription(void *p);
   static void destruct_ddastoyscLcLPulseDescription(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::PulseDescription*)
   {
      ::ddastoys::PulseDescription *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::ddastoys::PulseDescription));
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::PulseDescription", "fit_extensions.h", 37,
                  typeid(::ddastoys::PulseDescription), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ddastoyscLcLPulseDescription_Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::PulseDescription) );
      instance.SetNew(&new_ddastoyscLcLPulseDescription);
      instance.SetNewArray(&newArray_ddastoyscLcLPulseDescription);
      instance.SetDelete(&delete_ddastoyscLcLPulseDescription);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLPulseDescription);
      instance.SetDestructor(&destruct_ddastoyscLcLPulseDescription);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::PulseDescription*)
   {
      return GenerateInitInstanceLocal((::ddastoys::PulseDescription*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::PulseDescription*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ddastoyscLcLPulseDescription_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::ddastoys::PulseDescription*)nullptr)->GetClass();
      ddastoyscLcLPulseDescription_TClassManip(theClass);
   return theClass;
   }

   static void ddastoyscLcLPulseDescription_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *ddastoyscLcLfit1Info_Dictionary();
   static void ddastoyscLcLfit1Info_TClassManip(TClass*);
   static void *new_ddastoyscLcLfit1Info(void *p = nullptr);
   static void *newArray_ddastoyscLcLfit1Info(Long_t size, void *p);
   static void delete_ddastoyscLcLfit1Info(void *p);
   static void deleteArray_ddastoyscLcLfit1Info(void *p);
   static void destruct_ddastoyscLcLfit1Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::fit1Info*)
   {
      ::ddastoys::fit1Info *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::ddastoys::fit1Info));
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::fit1Info", "fit_extensions.h", 48,
                  typeid(::ddastoys::fit1Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ddastoyscLcLfit1Info_Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::fit1Info) );
      instance.SetNew(&new_ddastoyscLcLfit1Info);
      instance.SetNewArray(&newArray_ddastoyscLcLfit1Info);
      instance.SetDelete(&delete_ddastoyscLcLfit1Info);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLfit1Info);
      instance.SetDestructor(&destruct_ddastoyscLcLfit1Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::fit1Info*)
   {
      return GenerateInitInstanceLocal((::ddastoys::fit1Info*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::fit1Info*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ddastoyscLcLfit1Info_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::ddastoys::fit1Info*)nullptr)->GetClass();
      ddastoyscLcLfit1Info_TClassManip(theClass);
   return theClass;
   }

   static void ddastoyscLcLfit1Info_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *ddastoyscLcLfit2Info_Dictionary();
   static void ddastoyscLcLfit2Info_TClassManip(TClass*);
   static void *new_ddastoyscLcLfit2Info(void *p = nullptr);
   static void *newArray_ddastoyscLcLfit2Info(Long_t size, void *p);
   static void delete_ddastoyscLcLfit2Info(void *p);
   static void deleteArray_ddastoyscLcLfit2Info(void *p);
   static void destruct_ddastoyscLcLfit2Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::fit2Info*)
   {
      ::ddastoys::fit2Info *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::ddastoys::fit2Info));
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::fit2Info", "fit_extensions.h", 60,
                  typeid(::ddastoys::fit2Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ddastoyscLcLfit2Info_Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::fit2Info) );
      instance.SetNew(&new_ddastoyscLcLfit2Info);
      instance.SetNewArray(&newArray_ddastoyscLcLfit2Info);
      instance.SetDelete(&delete_ddastoyscLcLfit2Info);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLfit2Info);
      instance.SetDestructor(&destruct_ddastoyscLcLfit2Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::fit2Info*)
   {
      return GenerateInitInstanceLocal((::ddastoys::fit2Info*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::fit2Info*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ddastoyscLcLfit2Info_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::ddastoys::fit2Info*)nullptr)->GetClass();
      ddastoyscLcLfit2Info_TClassManip(theClass);
   return theClass;
   }

   static void ddastoyscLcLfit2Info_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *ddastoyscLcLHitExtension_Dictionary();
   static void ddastoyscLcLHitExtension_TClassManip(TClass*);
   static void *new_ddastoyscLcLHitExtension(void *p = nullptr);
   static void *newArray_ddastoyscLcLHitExtension(Long_t size, void *p);
   static void delete_ddastoyscLcLHitExtension(void *p);
   static void deleteArray_ddastoyscLcLHitExtension(void *p);
   static void destruct_ddastoyscLcLHitExtension(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::HitExtension*)
   {
      ::ddastoys::HitExtension *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::ddastoys::HitExtension));
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::HitExtension", "fit_extensions.h", 82,
                  typeid(::ddastoys::HitExtension), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ddastoyscLcLHitExtension_Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::HitExtension) );
      instance.SetNew(&new_ddastoyscLcLHitExtension);
      instance.SetNewArray(&newArray_ddastoyscLcLHitExtension);
      instance.SetDelete(&delete_ddastoyscLcLHitExtension);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLHitExtension);
      instance.SetDestructor(&destruct_ddastoyscLcLHitExtension);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::HitExtension*)
   {
      return GenerateInitInstanceLocal((::ddastoys::HitExtension*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::HitExtension*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ddastoyscLcLHitExtension_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::ddastoys::HitExtension*)nullptr)->GetClass();
      ddastoyscLcLHitExtension_TClassManip(theClass);
   return theClass;
   }

   static void ddastoyscLcLHitExtension_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *ddastoyscLcLDDASFitHit_Dictionary();
   static void ddastoyscLcLDDASFitHit_TClassManip(TClass*);
   static void *new_ddastoyscLcLDDASFitHit(void *p = nullptr);
   static void *newArray_ddastoyscLcLDDASFitHit(Long_t size, void *p);
   static void delete_ddastoyscLcLDDASFitHit(void *p);
   static void deleteArray_ddastoyscLcLDDASFitHit(void *p);
   static void destruct_ddastoyscLcLDDASFitHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::DDASFitHit*)
   {
      ::ddastoys::DDASFitHit *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::ddastoys::DDASFitHit));
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::DDASFitHit", "DDASFitHit.h", 45,
                  typeid(::ddastoys::DDASFitHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ddastoyscLcLDDASFitHit_Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::DDASFitHit) );
      instance.SetNew(&new_ddastoyscLcLDDASFitHit);
      instance.SetNewArray(&newArray_ddastoyscLcLDDASFitHit);
      instance.SetDelete(&delete_ddastoyscLcLDDASFitHit);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLDDASFitHit);
      instance.SetDestructor(&destruct_ddastoyscLcLDDASFitHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::DDASFitHit*)
   {
      return GenerateInitInstanceLocal((::ddastoys::DDASFitHit*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::DDASFitHit*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ddastoyscLcLDDASFitHit_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASFitHit*)nullptr)->GetClass();
      ddastoyscLcLDDASFitHit_TClassManip(theClass);
   return theClass;
   }

   static void ddastoyscLcLDDASFitHit_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static void *new_ddastoyscLcLDDASRootFitHit(void *p = nullptr);
   static void *newArray_ddastoyscLcLDDASRootFitHit(Long_t size, void *p);
   static void delete_ddastoyscLcLDDASRootFitHit(void *p);
   static void deleteArray_ddastoyscLcLDDASRootFitHit(void *p);
   static void destruct_ddastoyscLcLDDASRootFitHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::DDASRootFitHit*)
   {
      ::ddastoys::DDASRootFitHit *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::DDASRootFitHit >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::DDASRootFitHit", ::ddastoys::DDASRootFitHit::Class_Version(), "DDASRootFitHit.h", 52,
                  typeid(::ddastoys::DDASRootFitHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::DDASRootFitHit::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::DDASRootFitHit) );
      instance.SetNew(&new_ddastoyscLcLDDASRootFitHit);
      instance.SetNewArray(&newArray_ddastoyscLcLDDASRootFitHit);
      instance.SetDelete(&delete_ddastoyscLcLDDASRootFitHit);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLDDASRootFitHit);
      instance.SetDestructor(&destruct_ddastoyscLcLDDASRootFitHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::DDASRootFitHit*)
   {
      return GenerateInitInstanceLocal((::ddastoys::DDASRootFitHit*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitHit*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_ddastoyscLcLRootPulseDescription(void *p = nullptr);
   static void *newArray_ddastoyscLcLRootPulseDescription(Long_t size, void *p);
   static void delete_ddastoyscLcLRootPulseDescription(void *p);
   static void deleteArray_ddastoyscLcLRootPulseDescription(void *p);
   static void destruct_ddastoyscLcLRootPulseDescription(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::RootPulseDescription*)
   {
      ::ddastoys::RootPulseDescription *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::RootPulseDescription >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::RootPulseDescription", ::ddastoys::RootPulseDescription::Class_Version(), "RootExtensions.h", 48,
                  typeid(::ddastoys::RootPulseDescription), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::RootPulseDescription::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::RootPulseDescription) );
      instance.SetNew(&new_ddastoyscLcLRootPulseDescription);
      instance.SetNewArray(&newArray_ddastoyscLcLRootPulseDescription);
      instance.SetDelete(&delete_ddastoyscLcLRootPulseDescription);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLRootPulseDescription);
      instance.SetDestructor(&destruct_ddastoyscLcLRootPulseDescription);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::RootPulseDescription*)
   {
      return GenerateInitInstanceLocal((::ddastoys::RootPulseDescription*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::RootPulseDescription*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_ddastoyscLcLRootFit1Info(void *p = nullptr);
   static void *newArray_ddastoyscLcLRootFit1Info(Long_t size, void *p);
   static void delete_ddastoyscLcLRootFit1Info(void *p);
   static void deleteArray_ddastoyscLcLRootFit1Info(void *p);
   static void destruct_ddastoyscLcLRootFit1Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::RootFit1Info*)
   {
      ::ddastoys::RootFit1Info *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::RootFit1Info >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::RootFit1Info", ::ddastoys::RootFit1Info::Class_Version(), "RootExtensions.h", 58,
                  typeid(::ddastoys::RootFit1Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::RootFit1Info::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::RootFit1Info) );
      instance.SetNew(&new_ddastoyscLcLRootFit1Info);
      instance.SetNewArray(&newArray_ddastoyscLcLRootFit1Info);
      instance.SetDelete(&delete_ddastoyscLcLRootFit1Info);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLRootFit1Info);
      instance.SetDestructor(&destruct_ddastoyscLcLRootFit1Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::RootFit1Info*)
   {
      return GenerateInitInstanceLocal((::ddastoys::RootFit1Info*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::RootFit1Info*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_ddastoyscLcLRootFit2Info(void *p = nullptr);
   static void *newArray_ddastoyscLcLRootFit2Info(Long_t size, void *p);
   static void delete_ddastoyscLcLRootFit2Info(void *p);
   static void deleteArray_ddastoyscLcLRootFit2Info(void *p);
   static void destruct_ddastoyscLcLRootFit2Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::RootFit2Info*)
   {
      ::ddastoys::RootFit2Info *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::RootFit2Info >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::RootFit2Info", ::ddastoys::RootFit2Info::Class_Version(), "RootExtensions.h", 68,
                  typeid(::ddastoys::RootFit2Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::RootFit2Info::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::RootFit2Info) );
      instance.SetNew(&new_ddastoyscLcLRootFit2Info);
      instance.SetNewArray(&newArray_ddastoyscLcLRootFit2Info);
      instance.SetDelete(&delete_ddastoyscLcLRootFit2Info);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLRootFit2Info);
      instance.SetDestructor(&destruct_ddastoyscLcLRootFit2Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::RootFit2Info*)
   {
      return GenerateInitInstanceLocal((::ddastoys::RootFit2Info*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::RootFit2Info*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_ddastoyscLcLRootHitExtension(void *p = nullptr);
   static void *newArray_ddastoyscLcLRootHitExtension(Long_t size, void *p);
   static void delete_ddastoyscLcLRootHitExtension(void *p);
   static void deleteArray_ddastoyscLcLRootHitExtension(void *p);
   static void destruct_ddastoyscLcLRootHitExtension(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::ddastoys::RootHitExtension*)
   {
      ::ddastoys::RootHitExtension *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::ddastoys::RootHitExtension >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("ddastoys::RootHitExtension", ::ddastoys::RootHitExtension::Class_Version(), "RootExtensions.h", 79,
                  typeid(::ddastoys::RootHitExtension), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::ddastoys::RootHitExtension::Dictionary, isa_proxy, 4,
                  sizeof(::ddastoys::RootHitExtension) );
      instance.SetNew(&new_ddastoyscLcLRootHitExtension);
      instance.SetNewArray(&newArray_ddastoyscLcLRootHitExtension);
      instance.SetDelete(&delete_ddastoyscLcLRootHitExtension);
      instance.SetDeleteArray(&deleteArray_ddastoyscLcLRootHitExtension);
      instance.SetDestructor(&destruct_ddastoyscLcLRootHitExtension);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::ddastoys::RootHitExtension*)
   {
      return GenerateInitInstanceLocal((::ddastoys::RootHitExtension*)nullptr);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::ddastoys::RootHitExtension*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr DDASRootFitEvent::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *DDASRootFitEvent::Class_Name()
{
   return "ddastoys::DDASRootFitEvent";
}

//______________________________________________________________________________
const char *DDASRootFitEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitEvent*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int DDASRootFitEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitEvent*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *DDASRootFitEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitEvent*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *DDASRootFitEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitEvent*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr DDASRootFitHit::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *DDASRootFitHit::Class_Name()
{
   return "ddastoys::DDASRootFitHit";
}

//______________________________________________________________________________
const char *DDASRootFitHit::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitHit*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int DDASRootFitHit::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitHit*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *DDASRootFitHit::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitHit*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *DDASRootFitHit::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::DDASRootFitHit*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr RootPulseDescription::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RootPulseDescription::Class_Name()
{
   return "ddastoys::RootPulseDescription";
}

//______________________________________________________________________________
const char *RootPulseDescription::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootPulseDescription*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RootPulseDescription::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootPulseDescription*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootPulseDescription::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootPulseDescription*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootPulseDescription::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootPulseDescription*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr RootFit1Info::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RootFit1Info::Class_Name()
{
   return "ddastoys::RootFit1Info";
}

//______________________________________________________________________________
const char *RootFit1Info::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit1Info*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RootFit1Info::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit1Info*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootFit1Info::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit1Info*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootFit1Info::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit1Info*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr RootFit2Info::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RootFit2Info::Class_Name()
{
   return "ddastoys::RootFit2Info";
}

//______________________________________________________________________________
const char *RootFit2Info::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit2Info*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RootFit2Info::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit2Info*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootFit2Info::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit2Info*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootFit2Info::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootFit2Info*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
atomic_TClass_ptr RootHitExtension::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RootHitExtension::Class_Name()
{
   return "ddastoys::RootHitExtension";
}

//______________________________________________________________________________
const char *RootHitExtension::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootHitExtension*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RootHitExtension::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootHitExtension*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootHitExtension::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootHitExtension*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootHitExtension::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::ddastoys::RootHitExtension*)nullptr)->GetClass(); }
   return fgIsA;
}

} // namespace ddastoys
namespace ddastoys {
//______________________________________________________________________________
void DDASRootFitEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::DDASRootFitEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::DDASRootFitEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::DDASRootFitEvent::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLDDASRootFitEvent(void *p) {
      return  p ? new(p) ::ddastoys::DDASRootFitEvent : new ::ddastoys::DDASRootFitEvent;
   }
   static void *newArray_ddastoyscLcLDDASRootFitEvent(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::DDASRootFitEvent[nElements] : new ::ddastoys::DDASRootFitEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLDDASRootFitEvent(void *p) {
      delete ((::ddastoys::DDASRootFitEvent*)p);
   }
   static void deleteArray_ddastoyscLcLDDASRootFitEvent(void *p) {
      delete [] ((::ddastoys::DDASRootFitEvent*)p);
   }
   static void destruct_ddastoyscLcLDDASRootFitEvent(void *p) {
      typedef ::ddastoys::DDASRootFitEvent current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::DDASRootFitEvent

namespace ROOT {
   // Wrappers around operator new
   static void *new_DAQcLcLDDAScLcLDDASHit(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::DAQ::DDAS::DDASHit : new ::DAQ::DDAS::DDASHit;
   }
   static void *newArray_DAQcLcLDDAScLcLDDASHit(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::DAQ::DDAS::DDASHit[nElements] : new ::DAQ::DDAS::DDASHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_DAQcLcLDDAScLcLDDASHit(void *p) {
      delete ((::DAQ::DDAS::DDASHit*)p);
   }
   static void deleteArray_DAQcLcLDDAScLcLDDASHit(void *p) {
      delete [] ((::DAQ::DDAS::DDASHit*)p);
   }
   static void destruct_DAQcLcLDDAScLcLDDASHit(void *p) {
      typedef ::DAQ::DDAS::DDASHit current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::DAQ::DDAS::DDASHit

namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLPulseDescription(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::PulseDescription : new ::ddastoys::PulseDescription;
   }
   static void *newArray_ddastoyscLcLPulseDescription(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::PulseDescription[nElements] : new ::ddastoys::PulseDescription[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLPulseDescription(void *p) {
      delete ((::ddastoys::PulseDescription*)p);
   }
   static void deleteArray_ddastoyscLcLPulseDescription(void *p) {
      delete [] ((::ddastoys::PulseDescription*)p);
   }
   static void destruct_ddastoyscLcLPulseDescription(void *p) {
      typedef ::ddastoys::PulseDescription current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::PulseDescription

namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLfit1Info(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::fit1Info : new ::ddastoys::fit1Info;
   }
   static void *newArray_ddastoyscLcLfit1Info(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::fit1Info[nElements] : new ::ddastoys::fit1Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLfit1Info(void *p) {
      delete ((::ddastoys::fit1Info*)p);
   }
   static void deleteArray_ddastoyscLcLfit1Info(void *p) {
      delete [] ((::ddastoys::fit1Info*)p);
   }
   static void destruct_ddastoyscLcLfit1Info(void *p) {
      typedef ::ddastoys::fit1Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::fit1Info

namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLfit2Info(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::fit2Info : new ::ddastoys::fit2Info;
   }
   static void *newArray_ddastoyscLcLfit2Info(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::fit2Info[nElements] : new ::ddastoys::fit2Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLfit2Info(void *p) {
      delete ((::ddastoys::fit2Info*)p);
   }
   static void deleteArray_ddastoyscLcLfit2Info(void *p) {
      delete [] ((::ddastoys::fit2Info*)p);
   }
   static void destruct_ddastoyscLcLfit2Info(void *p) {
      typedef ::ddastoys::fit2Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::fit2Info

namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLHitExtension(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::HitExtension : new ::ddastoys::HitExtension;
   }
   static void *newArray_ddastoyscLcLHitExtension(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::HitExtension[nElements] : new ::ddastoys::HitExtension[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLHitExtension(void *p) {
      delete ((::ddastoys::HitExtension*)p);
   }
   static void deleteArray_ddastoyscLcLHitExtension(void *p) {
      delete [] ((::ddastoys::HitExtension*)p);
   }
   static void destruct_ddastoyscLcLHitExtension(void *p) {
      typedef ::ddastoys::HitExtension current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::HitExtension

namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLDDASFitHit(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::DDASFitHit : new ::ddastoys::DDASFitHit;
   }
   static void *newArray_ddastoyscLcLDDASFitHit(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::ddastoys::DDASFitHit[nElements] : new ::ddastoys::DDASFitHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLDDASFitHit(void *p) {
      delete ((::ddastoys::DDASFitHit*)p);
   }
   static void deleteArray_ddastoyscLcLDDASFitHit(void *p) {
      delete [] ((::ddastoys::DDASFitHit*)p);
   }
   static void destruct_ddastoyscLcLDDASFitHit(void *p) {
      typedef ::ddastoys::DDASFitHit current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::DDASFitHit

namespace ddastoys {
//______________________________________________________________________________
void DDASRootFitHit::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::DDASRootFitHit.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::DDASRootFitHit::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::DDASRootFitHit::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLDDASRootFitHit(void *p) {
      return  p ? new(p) ::ddastoys::DDASRootFitHit : new ::ddastoys::DDASRootFitHit;
   }
   static void *newArray_ddastoyscLcLDDASRootFitHit(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::DDASRootFitHit[nElements] : new ::ddastoys::DDASRootFitHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLDDASRootFitHit(void *p) {
      delete ((::ddastoys::DDASRootFitHit*)p);
   }
   static void deleteArray_ddastoyscLcLDDASRootFitHit(void *p) {
      delete [] ((::ddastoys::DDASRootFitHit*)p);
   }
   static void destruct_ddastoyscLcLDDASRootFitHit(void *p) {
      typedef ::ddastoys::DDASRootFitHit current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::DDASRootFitHit

namespace ddastoys {
//______________________________________________________________________________
void RootPulseDescription::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::RootPulseDescription.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::RootPulseDescription::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::RootPulseDescription::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLRootPulseDescription(void *p) {
      return  p ? new(p) ::ddastoys::RootPulseDescription : new ::ddastoys::RootPulseDescription;
   }
   static void *newArray_ddastoyscLcLRootPulseDescription(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::RootPulseDescription[nElements] : new ::ddastoys::RootPulseDescription[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLRootPulseDescription(void *p) {
      delete ((::ddastoys::RootPulseDescription*)p);
   }
   static void deleteArray_ddastoyscLcLRootPulseDescription(void *p) {
      delete [] ((::ddastoys::RootPulseDescription*)p);
   }
   static void destruct_ddastoyscLcLRootPulseDescription(void *p) {
      typedef ::ddastoys::RootPulseDescription current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::RootPulseDescription

namespace ddastoys {
//______________________________________________________________________________
void RootFit1Info::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::RootFit1Info.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::RootFit1Info::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::RootFit1Info::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLRootFit1Info(void *p) {
      return  p ? new(p) ::ddastoys::RootFit1Info : new ::ddastoys::RootFit1Info;
   }
   static void *newArray_ddastoyscLcLRootFit1Info(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::RootFit1Info[nElements] : new ::ddastoys::RootFit1Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLRootFit1Info(void *p) {
      delete ((::ddastoys::RootFit1Info*)p);
   }
   static void deleteArray_ddastoyscLcLRootFit1Info(void *p) {
      delete [] ((::ddastoys::RootFit1Info*)p);
   }
   static void destruct_ddastoyscLcLRootFit1Info(void *p) {
      typedef ::ddastoys::RootFit1Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::RootFit1Info

namespace ddastoys {
//______________________________________________________________________________
void RootFit2Info::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::RootFit2Info.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::RootFit2Info::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::RootFit2Info::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLRootFit2Info(void *p) {
      return  p ? new(p) ::ddastoys::RootFit2Info : new ::ddastoys::RootFit2Info;
   }
   static void *newArray_ddastoyscLcLRootFit2Info(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::RootFit2Info[nElements] : new ::ddastoys::RootFit2Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLRootFit2Info(void *p) {
      delete ((::ddastoys::RootFit2Info*)p);
   }
   static void deleteArray_ddastoyscLcLRootFit2Info(void *p) {
      delete [] ((::ddastoys::RootFit2Info*)p);
   }
   static void destruct_ddastoyscLcLRootFit2Info(void *p) {
      typedef ::ddastoys::RootFit2Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::RootFit2Info

namespace ddastoys {
//______________________________________________________________________________
void RootHitExtension::Streamer(TBuffer &R__b)
{
   // Stream an object of class ddastoys::RootHitExtension.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ddastoys::RootHitExtension::Class(),this);
   } else {
      R__b.WriteClassBuffer(ddastoys::RootHitExtension::Class(),this);
   }
}

} // namespace ddastoys
namespace ROOT {
   // Wrappers around operator new
   static void *new_ddastoyscLcLRootHitExtension(void *p) {
      return  p ? new(p) ::ddastoys::RootHitExtension : new ::ddastoys::RootHitExtension;
   }
   static void *newArray_ddastoyscLcLRootHitExtension(Long_t nElements, void *p) {
      return p ? new(p) ::ddastoys::RootHitExtension[nElements] : new ::ddastoys::RootHitExtension[nElements];
   }
   // Wrapper around operator delete
   static void delete_ddastoyscLcLRootHitExtension(void *p) {
      delete ((::ddastoys::RootHitExtension*)p);
   }
   static void deleteArray_ddastoyscLcLRootHitExtension(void *p) {
      delete [] ((::ddastoys::RootHitExtension*)p);
   }
   static void destruct_ddastoyscLcLRootHitExtension(void *p) {
      typedef ::ddastoys::RootHitExtension current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::ddastoys::RootHitExtension

namespace ROOT {
   static TClass *vectorlEunsignedsPshortgR_Dictionary();
   static void vectorlEunsignedsPshortgR_TClassManip(TClass*);
   static void *new_vectorlEunsignedsPshortgR(void *p = nullptr);
   static void *newArray_vectorlEunsignedsPshortgR(Long_t size, void *p);
   static void delete_vectorlEunsignedsPshortgR(void *p);
   static void deleteArray_vectorlEunsignedsPshortgR(void *p);
   static void destruct_vectorlEunsignedsPshortgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<unsigned short>*)
   {
      vector<unsigned short> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<unsigned short>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<unsigned short>", -2, "vector", 389,
                  typeid(vector<unsigned short>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEunsignedsPshortgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<unsigned short>) );
      instance.SetNew(&new_vectorlEunsignedsPshortgR);
      instance.SetNewArray(&newArray_vectorlEunsignedsPshortgR);
      instance.SetDelete(&delete_vectorlEunsignedsPshortgR);
      instance.SetDeleteArray(&deleteArray_vectorlEunsignedsPshortgR);
      instance.SetDestructor(&destruct_vectorlEunsignedsPshortgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<unsigned short> >()));

      ::ROOT::AddClassAlternate("vector<unsigned short>","std::vector<unsigned short, std::allocator<unsigned short> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<unsigned short>*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEunsignedsPshortgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<unsigned short>*)nullptr)->GetClass();
      vectorlEunsignedsPshortgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEunsignedsPshortgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEunsignedsPshortgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<unsigned short> : new vector<unsigned short>;
   }
   static void *newArray_vectorlEunsignedsPshortgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<unsigned short>[nElements] : new vector<unsigned short>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEunsignedsPshortgR(void *p) {
      delete ((vector<unsigned short>*)p);
   }
   static void deleteArray_vectorlEunsignedsPshortgR(void *p) {
      delete [] ((vector<unsigned short>*)p);
   }
   static void destruct_vectorlEunsignedsPshortgR(void *p) {
      typedef vector<unsigned short> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<unsigned short>

namespace ROOT {
   static TClass *vectorlEunsignedsPintgR_Dictionary();
   static void vectorlEunsignedsPintgR_TClassManip(TClass*);
   static void *new_vectorlEunsignedsPintgR(void *p = nullptr);
   static void *newArray_vectorlEunsignedsPintgR(Long_t size, void *p);
   static void delete_vectorlEunsignedsPintgR(void *p);
   static void deleteArray_vectorlEunsignedsPintgR(void *p);
   static void destruct_vectorlEunsignedsPintgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<unsigned int>*)
   {
      vector<unsigned int> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<unsigned int>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<unsigned int>", -2, "vector", 389,
                  typeid(vector<unsigned int>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEunsignedsPintgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<unsigned int>) );
      instance.SetNew(&new_vectorlEunsignedsPintgR);
      instance.SetNewArray(&newArray_vectorlEunsignedsPintgR);
      instance.SetDelete(&delete_vectorlEunsignedsPintgR);
      instance.SetDeleteArray(&deleteArray_vectorlEunsignedsPintgR);
      instance.SetDestructor(&destruct_vectorlEunsignedsPintgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<unsigned int> >()));

      ::ROOT::AddClassAlternate("vector<unsigned int>","std::vector<unsigned int, std::allocator<unsigned int> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<unsigned int>*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEunsignedsPintgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<unsigned int>*)nullptr)->GetClass();
      vectorlEunsignedsPintgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEunsignedsPintgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEunsignedsPintgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<unsigned int> : new vector<unsigned int>;
   }
   static void *newArray_vectorlEunsignedsPintgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<unsigned int>[nElements] : new vector<unsigned int>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEunsignedsPintgR(void *p) {
      delete ((vector<unsigned int>*)p);
   }
   static void deleteArray_vectorlEunsignedsPintgR(void *p) {
      delete [] ((vector<unsigned int>*)p);
   }
   static void destruct_vectorlEunsignedsPintgR(void *p) {
      typedef vector<unsigned int> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<unsigned int>

namespace ROOT {
   static TClass *vectorlEddastoyscLcLRootHitExtensiongR_Dictionary();
   static void vectorlEddastoyscLcLRootHitExtensiongR_TClassManip(TClass*);
   static void *new_vectorlEddastoyscLcLRootHitExtensiongR(void *p = nullptr);
   static void *newArray_vectorlEddastoyscLcLRootHitExtensiongR(Long_t size, void *p);
   static void delete_vectorlEddastoyscLcLRootHitExtensiongR(void *p);
   static void deleteArray_vectorlEddastoyscLcLRootHitExtensiongR(void *p);
   static void destruct_vectorlEddastoyscLcLRootHitExtensiongR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<ddastoys::RootHitExtension>*)
   {
      vector<ddastoys::RootHitExtension> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<ddastoys::RootHitExtension>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<ddastoys::RootHitExtension>", -2, "vector", 389,
                  typeid(vector<ddastoys::RootHitExtension>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEddastoyscLcLRootHitExtensiongR_Dictionary, isa_proxy, 4,
                  sizeof(vector<ddastoys::RootHitExtension>) );
      instance.SetNew(&new_vectorlEddastoyscLcLRootHitExtensiongR);
      instance.SetNewArray(&newArray_vectorlEddastoyscLcLRootHitExtensiongR);
      instance.SetDelete(&delete_vectorlEddastoyscLcLRootHitExtensiongR);
      instance.SetDeleteArray(&deleteArray_vectorlEddastoyscLcLRootHitExtensiongR);
      instance.SetDestructor(&destruct_vectorlEddastoyscLcLRootHitExtensiongR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<ddastoys::RootHitExtension> >()));

      ::ROOT::AddClassAlternate("vector<ddastoys::RootHitExtension>","std::vector<ddastoys::RootHitExtension, std::allocator<ddastoys::RootHitExtension> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<ddastoys::RootHitExtension>*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEddastoyscLcLRootHitExtensiongR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<ddastoys::RootHitExtension>*)nullptr)->GetClass();
      vectorlEddastoyscLcLRootHitExtensiongR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEddastoyscLcLRootHitExtensiongR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEddastoyscLcLRootHitExtensiongR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<ddastoys::RootHitExtension> : new vector<ddastoys::RootHitExtension>;
   }
   static void *newArray_vectorlEddastoyscLcLRootHitExtensiongR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<ddastoys::RootHitExtension>[nElements] : new vector<ddastoys::RootHitExtension>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEddastoyscLcLRootHitExtensiongR(void *p) {
      delete ((vector<ddastoys::RootHitExtension>*)p);
   }
   static void deleteArray_vectorlEddastoyscLcLRootHitExtensiongR(void *p) {
      delete [] ((vector<ddastoys::RootHitExtension>*)p);
   }
   static void destruct_vectorlEddastoyscLcLRootHitExtensiongR(void *p) {
      typedef vector<ddastoys::RootHitExtension> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<ddastoys::RootHitExtension>

namespace ROOT {
   static TClass *vectorlEddastoyscLcLDDASRootFitHitmUgR_Dictionary();
   static void vectorlEddastoyscLcLDDASRootFitHitmUgR_TClassManip(TClass*);
   static void *new_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p = nullptr);
   static void *newArray_vectorlEddastoyscLcLDDASRootFitHitmUgR(Long_t size, void *p);
   static void delete_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p);
   static void deleteArray_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p);
   static void destruct_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<ddastoys::DDASRootFitHit*>*)
   {
      vector<ddastoys::DDASRootFitHit*> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<ddastoys::DDASRootFitHit*>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<ddastoys::DDASRootFitHit*>", -2, "vector", 389,
                  typeid(vector<ddastoys::DDASRootFitHit*>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEddastoyscLcLDDASRootFitHitmUgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<ddastoys::DDASRootFitHit*>) );
      instance.SetNew(&new_vectorlEddastoyscLcLDDASRootFitHitmUgR);
      instance.SetNewArray(&newArray_vectorlEddastoyscLcLDDASRootFitHitmUgR);
      instance.SetDelete(&delete_vectorlEddastoyscLcLDDASRootFitHitmUgR);
      instance.SetDeleteArray(&deleteArray_vectorlEddastoyscLcLDDASRootFitHitmUgR);
      instance.SetDestructor(&destruct_vectorlEddastoyscLcLDDASRootFitHitmUgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<ddastoys::DDASRootFitHit*> >()));

      ::ROOT::AddClassAlternate("vector<ddastoys::DDASRootFitHit*>","std::vector<ddastoys::DDASRootFitHit*, std::allocator<ddastoys::DDASRootFitHit*> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<ddastoys::DDASRootFitHit*>*)nullptr); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEddastoyscLcLDDASRootFitHitmUgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<ddastoys::DDASRootFitHit*>*)nullptr)->GetClass();
      vectorlEddastoyscLcLDDASRootFitHitmUgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEddastoyscLcLDDASRootFitHitmUgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<ddastoys::DDASRootFitHit*> : new vector<ddastoys::DDASRootFitHit*>;
   }
   static void *newArray_vectorlEddastoyscLcLDDASRootFitHitmUgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<ddastoys::DDASRootFitHit*>[nElements] : new vector<ddastoys::DDASRootFitHit*>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p) {
      delete ((vector<ddastoys::DDASRootFitHit*>*)p);
   }
   static void deleteArray_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p) {
      delete [] ((vector<ddastoys::DDASRootFitHit*>*)p);
   }
   static void destruct_vectorlEddastoyscLcLDDASRootFitHitmUgR(void *p) {
      typedef vector<ddastoys::DDASRootFitHit*> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<ddastoys::DDASRootFitHit*>

namespace {
  void TriggerDictionaryInitialization_DDASRootFitFormat_Impl() {
    static const char* headers[] = {
"DDASRootFitEvent.h",
"DDASRootFitHit.h",
"../DDASFitHit.h",
"RootExtensions.h",
nullptr
    };
    static const char* includePaths[] = {
"..",
"/usr/opt/ddastoys/6.0-000/DDASFormat/include",
"/usr/opt/root/6.26.04/include/",
"/aaron/DDASToys/EEConverter/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "DDASRootFitFormat dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootHitExtension;}
namespace std{template <typename _Tp> class __attribute__((annotate("$clingAutoload$bits/allocator.h")))  __attribute__((annotate("$clingAutoload$string")))  allocator;
}
namespace ddastoys{class __attribute__((annotate("$clingAutoload$DDASRootFitEvent.h")))  DDASRootFitEvent;}
namespace DAQ{namespace DDAS{class __attribute__((annotate("$clingAutoload$DDASHit.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  DDASHit;}}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$fit_extensions.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  PulseDescription;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$fit_extensions.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  fit1Info;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$fit_extensions.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  fit2Info;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$fit_extensions.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  HitExtension;}
namespace ddastoys{class __attribute__((annotate("$clingAutoload$DDASFitHit.h")))  __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  DDASFitHit;}
namespace ddastoys{class __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  DDASRootFitHit;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootPulseDescription;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootFit1Info;}
namespace ddastoys{struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootFit2Info;}
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "DDASRootFitFormat dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include "../DDASFitHit.h"
#include "RootExtensions.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"DAQ::DDAS::DDASHit", payloadCode, "@",
"ddastoys::DDASFitHit", payloadCode, "@",
"ddastoys::DDASRootFitEvent", payloadCode, "@",
"ddastoys::DDASRootFitHit", payloadCode, "@",
"ddastoys::HitExtension", payloadCode, "@",
"ddastoys::PulseDescription", payloadCode, "@",
"ddastoys::RootFit1Info", payloadCode, "@",
"ddastoys::RootFit2Info", payloadCode, "@",
"ddastoys::RootHitExtension", payloadCode, "@",
"ddastoys::RootPulseDescription", payloadCode, "@",
"ddastoys::fit1Info", payloadCode, "@",
"ddastoys::fit2Info", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("DDASRootFitFormat",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_DDASRootFitFormat_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_DDASRootFitFormat_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_DDASRootFitFormat() {
  TriggerDictionaryInitialization_DDASRootFitFormat_Impl();
}
