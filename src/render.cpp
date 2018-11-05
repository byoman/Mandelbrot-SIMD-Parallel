#include "render.hpp"
#include <cstdint>
#include <cassert>
#include <iostream>
#include <complex>
#include <SDL2/SDL.h>
#include <vector>
#include <xmmintrin.h>
#include <immintrin.h>
#include <algorithm>
#include <smmintrin.h>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

struct rgb8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

rgb8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgb8_t{0, g, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgb8_t{0, 255, b};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgb8_t{r, 255, 0};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgb8_t{255, b, 0};
  }
}

void print_vect(__m256 addToIters){

        __m256i tmpAddToIters = _mm256_cvtps_epi32(addToIters);
	std::cout << "addToIters = {" <<  _mm256_extract_epi32(tmpAddToIters, 0) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 1) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 2) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 3) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 4) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 5) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 6) << ", " 
          << _mm256_extract_epi32(tmpAddToIters, 7) << ", " << "}" << std::endl;
}

void compute_iters(int* nb_iters, int height, int roundedWidth, int width, int n_iterations, int start, int stop) {
  float Height = static_cast<float>(height);
  float Width  = static_cast<float>(width);
  __m256 incr = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
  __m256 vect0 = _mm256_set1_ps(0);
  __m256 vect1 = _mm256_set1_ps(1.0);
  __m256 vect2 = _mm256_set1_ps(2.0);
  __m256 vect25 = _mm256_set1_ps(2.5);
  __m256 vect025 = _mm256_set1_ps(0.25);
  __m256 vect4 = _mm256_set1_ps(4.0);
  __m256 vect100 = _mm256_set1_ps(n_iterations);
  __m256 scaleY = _mm256_set1_ps(2.0 / (Height - 1.0));
  __m256 scaleX = _mm256_set1_ps(3.5 /  (Width - 1.0));

  for (int i = start; i < (height / 2 + (height % 2)) * roundedWidth && i < stop; i+=8) {
     
    int y = i / roundedWidth;
    int x = i - (y * roundedWidth);

    float fx = static_cast<float>(x);
    float addrY[] = {static_cast<float>(y)};
    __m256 Ys = _mm256_broadcast_ss(addrY);
    __m256 Xs = _mm256_add_ps(_mm256_broadcast_ss(&fx), incr);

    
    __m256 Y0s = _mm256_fmsub_ps(scaleY, Ys, vect1);
    __m256 X0s = _mm256_fmsub_ps(scaleX, Xs, vect25);
    __m256 Xlocals = vect0;
    __m256 Ylocals = vect0;
    __m256 iters = vect0;
    __m256 tests = vect0;
    __m256 addToIters;
    __m256 testIters;

    __m256 XQs = _mm256_sub_ps(X0s, vect025);
    __m256 q = _mm256_add_ps(_mm256_mul_ps(XQs, XQs), _mm256_mul_ps(Y0s, Y0s));
    __m256 testQ = _mm256_mul_ps(q, _mm256_add_ps(q, XQs));
    testQ = _mm256_cmp_ps(testQ, _mm256_mul_ps(vect025, _mm256_mul_ps(Ys, Ys)), _CMP_LT_OQ);
    testQ = _mm256_and_ps(testQ, vect100);

    __m256 xb = _mm256_add_ps(X0s, vect1);
    __m256 b = _mm256_add_ps(_mm256_mul_ps(xb, xb), _mm256_mul_ps(Y0s, Y0s));
    __m256 testbulb = _mm256_cmp_ps(b, _mm256_set1_ps(0.0625), _CMP_LT_OQ);
    testbulb = _mm256_and_ps(testbulb, testQ);
    testbulb = _mm256_and_ps(testbulb, vect100);
    iters = testbulb;

    int iter = 0;
    int final_test;
    do {
      __m256 xSquare = _mm256_mul_ps(Xlocals, Xlocals);
      __m256 ySquare = _mm256_mul_ps(Ylocals, Ylocals);

      __m256 xTemp = _mm256_add_ps(_mm256_sub_ps(xSquare, ySquare), X0s);
      __m256 yTemp = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(Xlocals, Ylocals), vect2), Y0s);
      __m256 periodicity = _mm256_cmp_ps(_mm256_cmp_ps(Xlocals, xTemp, _CMP_EQ_OQ), _mm256_cmp_ps(Ylocals, yTemp, _CMP_EQ_OQ), _CMP_EQ_OQ);
      periodicity = _mm256_and_ps(periodicity, vect100);
      addToIters = _mm256_add_ps(addToIters, periodicity);

      Ylocals = yTemp;
      Xlocals = xTemp;

      tests = _mm256_add_ps(_mm256_mul_ps(Xlocals, Xlocals),_mm256_mul_ps(Ylocals, Ylocals)); // y² + x² 
      tests = _mm256_cmp_ps(tests, vect4, _CMP_LT_OQ); 
      testIters = _mm256_cmp_ps(iters, vect100, _CMP_LT_OQ);
      __m256 bothTests = _mm256_and_ps(tests, testIters);

      addToIters = _mm256_and_ps(bothTests, vect1);
      iters = _mm256_add_ps(iters, addToIters);

      __m256 tmp = _mm256_cmp_ps(addToIters, vect0, _CMP_EQ_OQ);
      final_test = _mm256_movemask_ps(tmp);
      iter++;

    } while((final_test != 0xFF) && iter < n_iterations);

    __m256i tmpIters = _mm256_cvtps_epi32(iters);
    _mm256_storeu_si256((__m256i*)(&(nb_iters[(y * roundedWidth) + x])), tmpIters);
    _mm256_storeu_si256((__m256i*)(&(nb_iters[((height - 1 - y) * roundedWidth) + x])), tmpIters);
  }

}

void render(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations)
{

  int roundedWidth = (width+7) & ~7UL;
  uint64_t total = width * height;
  std::vector<uint64_t> histogram;
  histogram.resize(n_iterations);
  int* nb_iters = new int[height * roundedWidth];
  std::vector<float> hue_vect;
  hue_vect.resize(n_iterations);

  compute_iters(nb_iters, height, roundedWidth, width, n_iterations, 0, (height / 2 + (height % 2)) * roundedWidth) ;
  for (int yn = 0; yn < height / 2 + (height % 2); yn++) {
    for (int x = 0; x < width; ++x) {
      histogram[nb_iters[(yn * roundedWidth) + x]]+=2;
    }
  }

  total -= histogram[n_iterations];
  float ftotal = static_cast<float>(total);
  hue_vect[0] = static_cast<float>(histogram[0]) / ftotal; 

  for (int i = 1; i < n_iterations - 1; i++){
    hue_vect[i] = (static_cast<float>(histogram[i]) / ftotal) + hue_vect[i - 1]; 
  }

  for (int y = 0; y < height; ++y) //display loop
  {
    rgb8_t* lineptr = reinterpret_cast<rgb8_t*>(buffer);

    for (int x = 0; x < width; ++x)
      if (nb_iters[(y * roundedWidth) + x] < n_iterations)
        lineptr[x] = heat_lut(hue_vect[nb_iters[(y * roundedWidth) + x]]);
    buffer += stride;
  }
}


void render_mt(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations)
{
  int roundedWidth = (width+7) & ~7UL;
  uint64_t total = width * height;
  std::vector<uint64_t> histogram;
  histogram.resize(n_iterations);
  int* nb_iters = new int[height * roundedWidth];
  std::vector<float> hue_vect;
  hue_vect.resize(n_iterations);



  auto lambda = [&](const tbb::blocked_range<int>& r) {
    compute_iters(nb_iters, height, roundedWidth, width, n_iterations, r.begin(), r.end());
  };
  int start = 0;
  int stop = roundedWidth * (height/2 + (height % 2));
  tbb::parallel_for(tbb::blocked_range<int>(start,stop), lambda); 

  for (int yn = 0; yn < height / 2; yn++) {
    for (int x = 0; x < width; ++x) {
      histogram[nb_iters[(yn * roundedWidth) + x]]+=2;
    }
  }

  total -= histogram[n_iterations];
  float ftotal = static_cast<float>(total);
  hue_vect[0] = static_cast<float>(histogram[0]) / ftotal; 

  for (int i = 1; i < n_iterations - 1; i++){
    hue_vect[i] = (static_cast<float>(histogram[i]) / ftotal) + hue_vect[i - 1]; 
  }

  for (int y = 0; y < height; ++y) //display loop
  {
    rgb8_t* lineptr = reinterpret_cast<rgb8_t*>(buffer);

    for (int x = 0; x < width; ++x)
      if (nb_iters[(y * roundedWidth) + x] < n_iterations)
        lineptr[x] = heat_lut(hue_vect[nb_iters[(y * roundedWidth) + x]]);
    buffer += stride;
  }

}
