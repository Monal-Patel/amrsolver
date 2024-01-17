/** \file maintest.cpp
 *  Entry point for unit tests
 */

#include <iostream>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <MMS.h>


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  // utest_env = new amr_wind_tests::AmrexTestEnv(argc, argv);
  // ::testing::AddGlobalTestEnvironment(utest_env);

  // initialise amrex
  Initialize(argc, argv, true, MPI_COMM_WORLD);

  return RUN_ALL_TESTS();
}