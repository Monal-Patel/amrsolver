#ifndef INDEX_H_
#define INDEX_H_

#include <AMReX_GpuContainers.H>

struct indicies_t {

public:
// if statement based selection on types of indicies
// Conserved variables

// Static constexpr not needed, but this allows compile time use of indicies if needed.
static constexpr int URHO=0;
static constexpr int UMX=1;
static constexpr int UMY=2;
static constexpr int UMZ=3;
static constexpr int UET=4;

// Primitive (derived) variables
static constexpr int QRHO=0;
static constexpr int QU=1;
static constexpr int QV=2;
static constexpr int QW=3;
static constexpr int QT=4;
static constexpr int QPRES=5;

static constexpr int NCONS=5;
static constexpr int NPRIM=6;
static constexpr int NGHOST=3; // TODO: make it an automatic parameter
};


struct indicies_multiphase_t {

public:
// Conserved (solved) variables
static constexpr int UR1=0;
static constexpr int UR2=1;
static constexpr int UMX=2;
static constexpr int UMY=3;
static constexpr int UMZ=4;
static constexpr int UET=5;
static constexpr int UA1=6;
static constexpr int NCONS=7;

// Primitive variables
static constexpr int QR1=0;
static constexpr int QR2=1;
static constexpr int QUX=2;
static constexpr int QUY=3;
static constexpr int QUZ=4;
static constexpr int QTP=5;
static constexpr int QPR=6;
static constexpr int QA1=7;
static constexpr int NPRIM=8;

static constexpr int NGHOST=3; // TODO: make it an automatic parameter
};

#endif