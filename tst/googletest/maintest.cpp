/** \file maintest.cpp
 *  Entry point for unit tests
 */

#include <iostream>
#include <gtest/gtest.h>
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  // utest_env = new amr_wind_tests::AmrexTestEnv(argc, argv);
  // ::testing::AddGlobalTestEnvironment(utest_env);



    std::cout << "hello" << std::endl;
  return RUN_ALL_TESTS();
}