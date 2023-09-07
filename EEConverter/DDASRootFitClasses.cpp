// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME DDASRootFitClasses
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
#include "RootExtensions.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_DDASRootFitEvent(void *p = 0);
   static void *newArray_DDASRootFitEvent(Long_t size, void *p);
   static void delete_DDASRootFitEvent(void *p);
   static void deleteArray_DDASRootFitEvent(void *p);
   static void destruct_DDASRootFitEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::DDASRootFitEvent*)
   {
      ::DDASRootFitEvent *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::DDASRootFitEvent >(0);
      static ::ROOT::TGenericClassInfo 
         instance("DDASRootFitEvent", ::DDASRootFitEvent::Class_Version(), "DDASRootFitEvent.h", 46,
                  typeid(::DDASRootFitEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::DDASRootFitEvent::Dictionary, isa_proxy, 4,
                  sizeof(::DDASRootFitEvent) );
      instance.SetNew(&new_DDASRootFitEvent);
      instance.SetNewArray(&newArray_DDASRootFitEvent);
      instance.SetDelete(&delete_DDASRootFitEvent);
      instance.SetDeleteArray(&deleteArray_DDASRootFitEvent);
      instance.SetDestructor(&destruct_DDASRootFitEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::DDASRootFitEvent*)
   {
      return GenerateInitInstanceLocal((::DDASRootFitEvent*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::DDASRootFitEvent*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_DDASRootFitHit(void *p = 0);
   static void *newArray_DDASRootFitHit(Long_t size, void *p);
   static void delete_DDASRootFitHit(void *p);
   static void deleteArray_DDASRootFitHit(void *p);
   static void destruct_DDASRootFitHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::DDASRootFitHit*)
   {
      ::DDASRootFitHit *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::DDASRootFitHit >(0);
      static ::ROOT::TGenericClassInfo 
         instance("DDASRootFitHit", ::DDASRootFitHit::Class_Version(), "DDASRootFitHit.h", 50,
                  typeid(::DDASRootFitHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::DDASRootFitHit::Dictionary, isa_proxy, 4,
                  sizeof(::DDASRootFitHit) );
      instance.SetNew(&new_DDASRootFitHit);
      instance.SetNewArray(&newArray_DDASRootFitHit);
      instance.SetDelete(&delete_DDASRootFitHit);
      instance.SetDeleteArray(&deleteArray_DDASRootFitHit);
      instance.SetDestructor(&destruct_DDASRootFitHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::DDASRootFitHit*)
   {
      return GenerateInitInstanceLocal((::DDASRootFitHit*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::DDASRootFitHit*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_RootPulseDescription(void *p = 0);
   static void *newArray_RootPulseDescription(Long_t size, void *p);
   static void delete_RootPulseDescription(void *p);
   static void deleteArray_RootPulseDescription(void *p);
   static void destruct_RootPulseDescription(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RootPulseDescription*)
   {
      ::RootPulseDescription *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RootPulseDescription >(0);
      static ::ROOT::TGenericClassInfo 
         instance("RootPulseDescription", ::RootPulseDescription::Class_Version(), "RootExtensions.h", 45,
                  typeid(::RootPulseDescription), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RootPulseDescription::Dictionary, isa_proxy, 4,
                  sizeof(::RootPulseDescription) );
      instance.SetNew(&new_RootPulseDescription);
      instance.SetNewArray(&newArray_RootPulseDescription);
      instance.SetDelete(&delete_RootPulseDescription);
      instance.SetDeleteArray(&deleteArray_RootPulseDescription);
      instance.SetDestructor(&destruct_RootPulseDescription);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RootPulseDescription*)
   {
      return GenerateInitInstanceLocal((::RootPulseDescription*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::RootPulseDescription*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_RootFit1Info(void *p = 0);
   static void *newArray_RootFit1Info(Long_t size, void *p);
   static void delete_RootFit1Info(void *p);
   static void deleteArray_RootFit1Info(void *p);
   static void destruct_RootFit1Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RootFit1Info*)
   {
      ::RootFit1Info *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RootFit1Info >(0);
      static ::ROOT::TGenericClassInfo 
         instance("RootFit1Info", ::RootFit1Info::Class_Version(), "RootExtensions.h", 75,
                  typeid(::RootFit1Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RootFit1Info::Dictionary, isa_proxy, 4,
                  sizeof(::RootFit1Info) );
      instance.SetNew(&new_RootFit1Info);
      instance.SetNewArray(&newArray_RootFit1Info);
      instance.SetDelete(&delete_RootFit1Info);
      instance.SetDeleteArray(&deleteArray_RootFit1Info);
      instance.SetDestructor(&destruct_RootFit1Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RootFit1Info*)
   {
      return GenerateInitInstanceLocal((::RootFit1Info*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::RootFit1Info*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_RootFit2Info(void *p = 0);
   static void *newArray_RootFit2Info(Long_t size, void *p);
   static void delete_RootFit2Info(void *p);
   static void deleteArray_RootFit2Info(void *p);
   static void destruct_RootFit2Info(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RootFit2Info*)
   {
      ::RootFit2Info *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RootFit2Info >(0);
      static ::ROOT::TGenericClassInfo 
         instance("RootFit2Info", ::RootFit2Info::Class_Version(), "RootExtensions.h", 104,
                  typeid(::RootFit2Info), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RootFit2Info::Dictionary, isa_proxy, 4,
                  sizeof(::RootFit2Info) );
      instance.SetNew(&new_RootFit2Info);
      instance.SetNewArray(&newArray_RootFit2Info);
      instance.SetDelete(&delete_RootFit2Info);
      instance.SetDeleteArray(&deleteArray_RootFit2Info);
      instance.SetDestructor(&destruct_RootFit2Info);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RootFit2Info*)
   {
      return GenerateInitInstanceLocal((::RootFit2Info*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::RootFit2Info*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_RootHitExtension(void *p = 0);
   static void *newArray_RootHitExtension(Long_t size, void *p);
   static void delete_RootHitExtension(void *p);
   static void deleteArray_RootHitExtension(void *p);
   static void destruct_RootHitExtension(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RootHitExtension*)
   {
      ::RootHitExtension *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RootHitExtension >(0);
      static ::ROOT::TGenericClassInfo 
         instance("RootHitExtension", ::RootHitExtension::Class_Version(), "RootExtensions.h", 133,
                  typeid(::RootHitExtension), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RootHitExtension::Dictionary, isa_proxy, 4,
                  sizeof(::RootHitExtension) );
      instance.SetNew(&new_RootHitExtension);
      instance.SetNewArray(&newArray_RootHitExtension);
      instance.SetDelete(&delete_RootHitExtension);
      instance.SetDeleteArray(&deleteArray_RootHitExtension);
      instance.SetDestructor(&destruct_RootHitExtension);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RootHitExtension*)
   {
      return GenerateInitInstanceLocal((::RootHitExtension*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const ::RootHitExtension*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr DDASRootFitEvent::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *DDASRootFitEvent::Class_Name()
{
   return "DDASRootFitEvent";
}

//______________________________________________________________________________
const char *DDASRootFitEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitEvent*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int DDASRootFitEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitEvent*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *DDASRootFitEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitEvent*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *DDASRootFitEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitEvent*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr DDASRootFitHit::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *DDASRootFitHit::Class_Name()
{
   return "DDASRootFitHit";
}

//______________________________________________________________________________
const char *DDASRootFitHit::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitHit*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int DDASRootFitHit::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitHit*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *DDASRootFitHit::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitHit*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *DDASRootFitHit::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::DDASRootFitHit*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr RootPulseDescription::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *RootPulseDescription::Class_Name()
{
   return "RootPulseDescription";
}

//______________________________________________________________________________
const char *RootPulseDescription::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootPulseDescription*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int RootPulseDescription::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootPulseDescription*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootPulseDescription::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootPulseDescription*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootPulseDescription::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootPulseDescription*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr RootFit1Info::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *RootFit1Info::Class_Name()
{
   return "RootFit1Info";
}

//______________________________________________________________________________
const char *RootFit1Info::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootFit1Info*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int RootFit1Info::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootFit1Info*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootFit1Info::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootFit1Info*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootFit1Info::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootFit1Info*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr RootFit2Info::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *RootFit2Info::Class_Name()
{
   return "RootFit2Info";
}

//______________________________________________________________________________
const char *RootFit2Info::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootFit2Info*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int RootFit2Info::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootFit2Info*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootFit2Info::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootFit2Info*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootFit2Info::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootFit2Info*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr RootHitExtension::fgIsA(0);  // static to hold class pointer

//______________________________________________________________________________
const char *RootHitExtension::Class_Name()
{
   return "RootHitExtension";
}

//______________________________________________________________________________
const char *RootHitExtension::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootHitExtension*)0x0)->GetImplFileName();
}

//______________________________________________________________________________
int RootHitExtension::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RootHitExtension*)0x0)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RootHitExtension::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootHitExtension*)0x0)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RootHitExtension::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RootHitExtension*)0x0)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void DDASRootFitEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class DDASRootFitEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(DDASRootFitEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(DDASRootFitEvent::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_DDASRootFitEvent(void *p) {
      return  p ? new(p) ::DDASRootFitEvent : new ::DDASRootFitEvent;
   }
   static void *newArray_DDASRootFitEvent(Long_t nElements, void *p) {
      return p ? new(p) ::DDASRootFitEvent[nElements] : new ::DDASRootFitEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_DDASRootFitEvent(void *p) {
      delete ((::DDASRootFitEvent*)p);
   }
   static void deleteArray_DDASRootFitEvent(void *p) {
      delete [] ((::DDASRootFitEvent*)p);
   }
   static void destruct_DDASRootFitEvent(void *p) {
      typedef ::DDASRootFitEvent current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::DDASRootFitEvent

//______________________________________________________________________________
void DDASRootFitHit::Streamer(TBuffer &R__b)
{
   // Stream an object of class DDASRootFitHit.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(DDASRootFitHit::Class(),this);
   } else {
      R__b.WriteClassBuffer(DDASRootFitHit::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_DDASRootFitHit(void *p) {
      return  p ? new(p) ::DDASRootFitHit : new ::DDASRootFitHit;
   }
   static void *newArray_DDASRootFitHit(Long_t nElements, void *p) {
      return p ? new(p) ::DDASRootFitHit[nElements] : new ::DDASRootFitHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_DDASRootFitHit(void *p) {
      delete ((::DDASRootFitHit*)p);
   }
   static void deleteArray_DDASRootFitHit(void *p) {
      delete [] ((::DDASRootFitHit*)p);
   }
   static void destruct_DDASRootFitHit(void *p) {
      typedef ::DDASRootFitHit current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::DDASRootFitHit

//______________________________________________________________________________
void RootPulseDescription::Streamer(TBuffer &R__b)
{
   // Stream an object of class RootPulseDescription.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RootPulseDescription::Class(),this);
   } else {
      R__b.WriteClassBuffer(RootPulseDescription::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RootPulseDescription(void *p) {
      return  p ? new(p) ::RootPulseDescription : new ::RootPulseDescription;
   }
   static void *newArray_RootPulseDescription(Long_t nElements, void *p) {
      return p ? new(p) ::RootPulseDescription[nElements] : new ::RootPulseDescription[nElements];
   }
   // Wrapper around operator delete
   static void delete_RootPulseDescription(void *p) {
      delete ((::RootPulseDescription*)p);
   }
   static void deleteArray_RootPulseDescription(void *p) {
      delete [] ((::RootPulseDescription*)p);
   }
   static void destruct_RootPulseDescription(void *p) {
      typedef ::RootPulseDescription current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::RootPulseDescription

//______________________________________________________________________________
void RootFit1Info::Streamer(TBuffer &R__b)
{
   // Stream an object of class RootFit1Info.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RootFit1Info::Class(),this);
   } else {
      R__b.WriteClassBuffer(RootFit1Info::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RootFit1Info(void *p) {
      return  p ? new(p) ::RootFit1Info : new ::RootFit1Info;
   }
   static void *newArray_RootFit1Info(Long_t nElements, void *p) {
      return p ? new(p) ::RootFit1Info[nElements] : new ::RootFit1Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_RootFit1Info(void *p) {
      delete ((::RootFit1Info*)p);
   }
   static void deleteArray_RootFit1Info(void *p) {
      delete [] ((::RootFit1Info*)p);
   }
   static void destruct_RootFit1Info(void *p) {
      typedef ::RootFit1Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::RootFit1Info

//______________________________________________________________________________
void RootFit2Info::Streamer(TBuffer &R__b)
{
   // Stream an object of class RootFit2Info.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RootFit2Info::Class(),this);
   } else {
      R__b.WriteClassBuffer(RootFit2Info::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RootFit2Info(void *p) {
      return  p ? new(p) ::RootFit2Info : new ::RootFit2Info;
   }
   static void *newArray_RootFit2Info(Long_t nElements, void *p) {
      return p ? new(p) ::RootFit2Info[nElements] : new ::RootFit2Info[nElements];
   }
   // Wrapper around operator delete
   static void delete_RootFit2Info(void *p) {
      delete ((::RootFit2Info*)p);
   }
   static void deleteArray_RootFit2Info(void *p) {
      delete [] ((::RootFit2Info*)p);
   }
   static void destruct_RootFit2Info(void *p) {
      typedef ::RootFit2Info current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::RootFit2Info

//______________________________________________________________________________
void RootHitExtension::Streamer(TBuffer &R__b)
{
   // Stream an object of class RootHitExtension.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RootHitExtension::Class(),this);
   } else {
      R__b.WriteClassBuffer(RootHitExtension::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RootHitExtension(void *p) {
      return  p ? new(p) ::RootHitExtension : new ::RootHitExtension;
   }
   static void *newArray_RootHitExtension(Long_t nElements, void *p) {
      return p ? new(p) ::RootHitExtension[nElements] : new ::RootHitExtension[nElements];
   }
   // Wrapper around operator delete
   static void delete_RootHitExtension(void *p) {
      delete ((::RootHitExtension*)p);
   }
   static void deleteArray_RootHitExtension(void *p) {
      delete [] ((::RootHitExtension*)p);
   }
   static void destruct_RootHitExtension(void *p) {
      typedef ::RootHitExtension current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::RootHitExtension

namespace ROOT {
   static TClass *vectorlEunsignedsPshortgR_Dictionary();
   static void vectorlEunsignedsPshortgR_TClassManip(TClass*);
   static void *new_vectorlEunsignedsPshortgR(void *p = 0);
   static void *newArray_vectorlEunsignedsPshortgR(Long_t size, void *p);
   static void delete_vectorlEunsignedsPshortgR(void *p);
   static void deleteArray_vectorlEunsignedsPshortgR(void *p);
   static void destruct_vectorlEunsignedsPshortgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<unsigned short>*)
   {
      vector<unsigned short> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<unsigned short>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<unsigned short>", -2, "vector", 339,
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
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<unsigned short>*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEunsignedsPshortgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<unsigned short>*)0x0)->GetClass();
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
   static void *new_vectorlEunsignedsPintgR(void *p = 0);
   static void *newArray_vectorlEunsignedsPintgR(Long_t size, void *p);
   static void delete_vectorlEunsignedsPintgR(void *p);
   static void deleteArray_vectorlEunsignedsPintgR(void *p);
   static void destruct_vectorlEunsignedsPintgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<unsigned int>*)
   {
      vector<unsigned int> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<unsigned int>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<unsigned int>", -2, "vector", 339,
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
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<unsigned int>*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEunsignedsPintgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<unsigned int>*)0x0)->GetClass();
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
   static TClass *vectorlERootHitExtensiongR_Dictionary();
   static void vectorlERootHitExtensiongR_TClassManip(TClass*);
   static void *new_vectorlERootHitExtensiongR(void *p = 0);
   static void *newArray_vectorlERootHitExtensiongR(Long_t size, void *p);
   static void delete_vectorlERootHitExtensiongR(void *p);
   static void deleteArray_vectorlERootHitExtensiongR(void *p);
   static void destruct_vectorlERootHitExtensiongR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<RootHitExtension>*)
   {
      vector<RootHitExtension> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<RootHitExtension>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<RootHitExtension>", -2, "vector", 339,
                  typeid(vector<RootHitExtension>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlERootHitExtensiongR_Dictionary, isa_proxy, 4,
                  sizeof(vector<RootHitExtension>) );
      instance.SetNew(&new_vectorlERootHitExtensiongR);
      instance.SetNewArray(&newArray_vectorlERootHitExtensiongR);
      instance.SetDelete(&delete_vectorlERootHitExtensiongR);
      instance.SetDeleteArray(&deleteArray_vectorlERootHitExtensiongR);
      instance.SetDestructor(&destruct_vectorlERootHitExtensiongR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<RootHitExtension> >()));

      ::ROOT::AddClassAlternate("vector<RootHitExtension>","std::vector<RootHitExtension, std::allocator<RootHitExtension> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<RootHitExtension>*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlERootHitExtensiongR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<RootHitExtension>*)0x0)->GetClass();
      vectorlERootHitExtensiongR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlERootHitExtensiongR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlERootHitExtensiongR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<RootHitExtension> : new vector<RootHitExtension>;
   }
   static void *newArray_vectorlERootHitExtensiongR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<RootHitExtension>[nElements] : new vector<RootHitExtension>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlERootHitExtensiongR(void *p) {
      delete ((vector<RootHitExtension>*)p);
   }
   static void deleteArray_vectorlERootHitExtensiongR(void *p) {
      delete [] ((vector<RootHitExtension>*)p);
   }
   static void destruct_vectorlERootHitExtensiongR(void *p) {
      typedef vector<RootHitExtension> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<RootHitExtension>

namespace ROOT {
   static TClass *vectorlEDDASRootFitHitmUgR_Dictionary();
   static void vectorlEDDASRootFitHitmUgR_TClassManip(TClass*);
   static void *new_vectorlEDDASRootFitHitmUgR(void *p = 0);
   static void *newArray_vectorlEDDASRootFitHitmUgR(Long_t size, void *p);
   static void delete_vectorlEDDASRootFitHitmUgR(void *p);
   static void deleteArray_vectorlEDDASRootFitHitmUgR(void *p);
   static void destruct_vectorlEDDASRootFitHitmUgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<DDASRootFitHit*>*)
   {
      vector<DDASRootFitHit*> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<DDASRootFitHit*>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<DDASRootFitHit*>", -2, "vector", 339,
                  typeid(vector<DDASRootFitHit*>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEDDASRootFitHitmUgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<DDASRootFitHit*>) );
      instance.SetNew(&new_vectorlEDDASRootFitHitmUgR);
      instance.SetNewArray(&newArray_vectorlEDDASRootFitHitmUgR);
      instance.SetDelete(&delete_vectorlEDDASRootFitHitmUgR);
      instance.SetDeleteArray(&deleteArray_vectorlEDDASRootFitHitmUgR);
      instance.SetDestructor(&destruct_vectorlEDDASRootFitHitmUgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<DDASRootFitHit*> >()));

      ::ROOT::AddClassAlternate("vector<DDASRootFitHit*>","std::vector<DDASRootFitHit*, std::allocator<DDASRootFitHit*> >");
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal((const vector<DDASRootFitHit*>*)0x0); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEDDASRootFitHitmUgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<DDASRootFitHit*>*)0x0)->GetClass();
      vectorlEDDASRootFitHitmUgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEDDASRootFitHitmUgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEDDASRootFitHitmUgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<DDASRootFitHit*> : new vector<DDASRootFitHit*>;
   }
   static void *newArray_vectorlEDDASRootFitHitmUgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<DDASRootFitHit*>[nElements] : new vector<DDASRootFitHit*>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEDDASRootFitHitmUgR(void *p) {
      delete ((vector<DDASRootFitHit*>*)p);
   }
   static void deleteArray_vectorlEDDASRootFitHitmUgR(void *p) {
      delete [] ((vector<DDASRootFitHit*>*)p);
   }
   static void destruct_vectorlEDDASRootFitHitmUgR(void *p) {
      typedef vector<DDASRootFitHit*> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<DDASRootFitHit*>

namespace {
  void TriggerDictionaryInitialization_DDASRootFitClasses_Impl() {
    static const char* headers[] = {
"DDASRootFitEvent.h",
"DDASRootFitHit.h",
"RootExtensions.h",
0
    };
    static const char* includePaths[] = {
"/usr/opt/root/root-6.24.06/include/",
"/aaron/devel/fdsi-analysis/ParallelAnalysis/EEConverter/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "DDASRootFitClasses dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootHitExtension;
namespace std{template <typename _Tp> class __attribute__((annotate("$clingAutoload$bits/allocator.h")))  __attribute__((annotate("$clingAutoload$string")))  allocator;
}
class __attribute__((annotate("$clingAutoload$DDASRootFitEvent.h")))  DDASRootFitEvent;
class __attribute__((annotate("$clingAutoload$DDASRootFitHit.h")))  DDASRootFitHit;
struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootPulseDescription;
struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootFit1Info;
struct __attribute__((annotate("$clingAutoload$RootExtensions.h")))  RootFit2Info;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "DDASRootFitClasses dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include "RootExtensions.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"DDASRootFitEvent", payloadCode, "@",
"DDASRootFitHit", payloadCode, "@",
"RootFit1Info", payloadCode, "@",
"RootFit2Info", payloadCode, "@",
"RootHitExtension", payloadCode, "@",
"RootPulseDescription", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("DDASRootFitClasses",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_DDASRootFitClasses_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_DDASRootFitClasses_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_DDASRootFitClasses() {
  TriggerDictionaryInitialization_DDASRootFitClasses_Impl();
}
