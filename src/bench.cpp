#include "render.hpp"
#include <benchmark/benchmark.h>

void BM_Rendering(benchmark::State& st)
{
  int height = 1080;
  int width = height * 16 / 9;
  int stride = width * 3;
  std::vector<std::uint8_t> data(height * width * 3);

  for (auto _ : st)
    render(reinterpret_cast<std::byte*>(data.data()), width, height, stride);

  st.SetItemsProcessed(st.iterations() * width * height);
}

void BM_Rendering_mt(benchmark::State& st)
{
  int height = 1080;
  int width = height * 16 / 9;
  int stride = width * 3;
  std::vector<std::uint8_t> data(height * width * 3);

  for (auto _ : st)
    render_mt(reinterpret_cast<std::byte*>(data.data()), width, height, stride);

  st.SetItemsProcessed(st.iterations() * width * height);
}

BENCHMARK(BM_Rendering)
->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Rendering_mt)
->Unit(benchmark::kMillisecond)
->UseRealTime();


// For development
// If you choose to keep the base and new versions
#if 0
void BM_Rendering_optimized(benchmark::State& st)
{
  int height = 1080;
  int width = height * 16 / 9;
  int stride = width * 3;
  std::vector<std::uint8_t> data(height * width * 3);

  for (auto _ : st)
    render_optimized(reinterpret_cast<std::byte*>(data.data()), width, height, stride);

  st.SetItemsProcessed(st.iterations() * width * height);
}

void BM_Rendering_optimized_mt(benchmark::State& st)
{
  int height = 1080;
  int width = height * 16 / 9;
  int stride = width * 3;
  std::vector<std::uint8_t> data(height * width * 3);

  for (auto _ : st)
    render_optimized_mt(reinterpret_cast<std::byte*>(data.data()), width, height, stride);

  st.SetItemsProcessed(st.iterations() * width * height);
}

BENCHMARK(BM_Rendering_optimized)
->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Rendering_optimized_mt)
->Unit(benchmark::kMillisecond)
->UseRealTime();

#endif

BENCHMARK_MAIN();
