////////////////////////////////////////////////////////////////////////////
//                                                                        //
// NOTICE TO USER:                                                        //
//                                                                        //
// This file is  part of the CUDA-PSO project which  provides a very fast //
// implementation of  Particle Swarm Optimization within  the nVIDIA CUDA //
// architecture.                                                          //
//                                                                        //
//                                                                        //
// Copyright (C) 2010 IbisLab, University of Parma (Italy)                //
// Authors: Luca Mussi, Fabio Daolio, Stefano Cagnoni                     //
//                                                                        //
//                                                                        //
// cudaPSO  is  free software:  you  can  redistribute  it  and/or modify //
// it under the  terms of the GNU General Public  License as published by //
// the Free Software Foundation, either  version 3 of the License, or (at //
// your option) any later version.                                        //
//                                                                        //
// cudaPSO  is distributed  in  the  hope that  it  will  be useful,  but //
// WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of //
// MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU //
// General Public License for more details.                               //
//                                                                        //
// You  should have received  a copy  of the  GNU General  Public License //
// along with cudaPSO. If not, see <http://www.gnu.org/licenses/>.        //
//                                                                        //
//                                                                        //
// If  you do  not agree  to these  terms, do  not download  or  use this //
// software.   For  any  inquiries  different from  license  information, //
// please contact                                                         //
//                                                                        //
// cagnoni@ce.unipr.it and mussi@ce.unipr.it                              //
//                                                                        //
// or                                                                     //
//                                                                        //
// Prof. Stefano Cagnoni / Luca Mussi                                     //
// IbisLab, Dipartimento di Ingegneria dell'Informazione                  //
// UniversitÓ degli Studi di Parma                                        //
// 181/a Viale G.P. Usberti                                               //
// I-43124 Parma                                                          //
// Italy                                                                  //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
#include <stdint.h>

#ifndef _REDUCTIONS_CU_
#define _REDUCTIONS_CU_

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

//*******************************
//  Parallel Reduction Operations
//*******************************
/// @brief inline kernel code for fast parallel reduction operations (max,
/// min, sum).

//! @brief Reduces an array of unsigned int elements to its minimum value.
//! @param vet Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <uint32_t  blockSize>
__device__ void reduceToMin(unsigned int* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    unsigned int mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = min(mySum, sdata[tid + 512]);
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = min(mySum, sdata[tid + 256]);
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = min(mySum, sdata[tid + 128]);
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = min(mySum, sdata[tid +  64]);
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile unsigned int* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = min(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = min(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = min(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = min(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = min(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile unsigned int* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = min(mySum, smem[tid + 32]); EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = min(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = min(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = min(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = min(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = min(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
}

//! @brief Reduces an array of float elements to its minimum value.
//! @param vet Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <uint32_t  blockSize>
__device__ void reduceToMin(float* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    float mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = fminf(mySum, sdata[tid + 512]);
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = fminf(mySum, sdata[tid + 256]);
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = fminf(mySum, sdata[tid + 128]);
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = fminf(mySum, sdata[tid +  64]);
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    //behavior.
	    volatile float* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = fminf(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile float* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = fminf(mySum, smem[tid + 32]); EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = fminf(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = fminf(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
}

//! @brief Reduces an array of unsigned int elements to its maximum value.
//! @param vet Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <uint32_t  blockSize>
__device__ void reduceToMax(unsigned int* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    unsigned int mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = max(mySum, sdata[tid + 512]);
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = max(mySum, sdata[tid + 256]);
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = max(mySum, sdata[tid + 128]);
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = max(mySum, sdata[tid +  64]);
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    //behavior.
	    volatile unsigned int* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = max(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = max(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = max(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = max(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = max(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile unsigned int* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = max(mySum, smem[tid + 32]); EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = max(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = max(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = max(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = max(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = max(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
}

//! @brief Reduces the elements of an array to their maximum value.
//! @param sdata Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <uint32_t  blockSize>
__device__ void reduceToMax(float* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    float mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 512]);
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 256]);
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 128]);
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = fmaxf(mySum, sdata[tid +  64]);
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile float* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile float* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid + 32]); EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); EMUSYNC;
	    }
	}
    }
}

//! @brief Reduces the elements of an array to their sum.
//! @param sdata Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <class T, uint32_t  blockSize>
__device__ void reduceToSum(T* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    T mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = mySum + sdata[tid + 512];
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = mySum + sdata[tid + 256];
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = mySum + sdata[tid + 128];
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = mySum + sdata[tid +  64];
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile T* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile T* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC;
	    }
	}
    }
}

//! @brief Reduces the elements of an array to their product.
//! @param sdata Pointer to shared memory data to be reduced.
//! @param tid Thread index.
template <class T, uint32_t  blockSize>
__device__ void reduceToProduct(T* sdata, uint32_t  tid){

    //Synchronize threads to share shared memory data
    __syncthreads();

    T mySum = sdata[tid];

    // do reduction in shared mem
    if (blockSize >= 1024) {
	if (tid < 512) {
	    sdata[tid] = mySum = mySum * sdata[tid + 512];
	} __syncthreads();
    }
    if (blockSize >= 512) {
	if (tid < 256) {
	    sdata[tid] = mySum = mySum * sdata[tid + 256];
	} __syncthreads();
    }
    if (blockSize >= 256) {
	if (tid < 128) {
	    sdata[tid] = mySum = mySum * sdata[tid + 128];
	} __syncthreads();
    }
    if (blockSize >= 128) {
	if (tid <  64) {
	    sdata[tid] = mySum = mySum * sdata[tid +  64];
	} __syncthreads();
    }

    if (blockSize == 32){
#ifndef __DEVICE_EMULATION__
	if (tid < 16)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile T* smem = sdata;
	    if (blockSize >=  32) {
		smem[tid] = mySum = mySum * smem[tid + 16]; EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = mySum * smem[tid +  8]; EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = mySum * smem[tid +  4]; EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = mySum * smem[tid +  2]; EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = mySum * smem[tid +  1]; EMUSYNC;
	    }
	}
    }
    else
    {
#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
	    // Now that we are using warp-synchronous programming (below)
	    // we need to declare our shared memory volatile so that the
	    // compiler doesn't reorder stores to it and induce incorrect
	    // behavior.
	    volatile T* smem = sdata;
	    if (blockSize >=  64) {
		smem[tid] = mySum = mySum * smem[tid + 32]; EMUSYNC;
	    }
	    if (blockSize >=  32) {
		smem[tid] = mySum = mySum * smem[tid + 16]; EMUSYNC;
	    }
	    if (blockSize >=  16) {
		smem[tid] = mySum = mySum * smem[tid +  8]; EMUSYNC;
	    }
	    if (blockSize >=   8) {
		smem[tid] = mySum = mySum * smem[tid +  4]; EMUSYNC;
	    }
	    if (blockSize >=   4) {
		smem[tid] = mySum = mySum * smem[tid +  2]; EMUSYNC;
	    }
	    if (blockSize >=   2) {
		smem[tid] = mySum = mySum * smem[tid +  1]; EMUSYNC;
	    }
	}
    }
}

#endif


