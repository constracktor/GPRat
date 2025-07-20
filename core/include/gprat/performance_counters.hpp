#ifndef GPRAT_PERFORMANCE_COUNTERS_HPP
#define GPRAT_PERFORMANCE_COUNTERS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <cstddef>
#include <cstdint>

GPRAT_NS_BEGIN

void track_tile_data_allocation(std::size_t size);
void track_tile_data_deallocation(std::size_t size);

void register_performance_counters();

GPRAT_NS_END

#endif
