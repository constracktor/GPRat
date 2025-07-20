#include "gprat/performance_counters.hpp"

#include <atomic>
#include <hpx/performance_counters/manage_counter_type.hpp>

GPRAT_NS_BEGIN

#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name)                                                                       \
    static std::atomic<std::uint64_t> name(0);                                                                         \
    std::uint64_t get_##name(bool reset) { return hpx::util::get_and_reset_value(name, reset); }

GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_allocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_deallocations)

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR

void track_tile_data_allocation(std::size_t /*size*/) { tile_data_allocations += 1; }

void track_tile_data_deallocation(std::size_t /*size*/) { tile_data_deallocations += 1; }

void register_performance_counters()
{
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_data/num_allocations",
        &get_tile_data_allocations,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_data/num_deallocations",
        &get_tile_data_deallocations,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
}

GPRAT_NS_END
