#include "gprat/performance_counters.hpp"

#include <atomic>
#include <hpx/util/get_and_reset_value.hpp>
#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
#include <hpx/performance_counters/manage_counter_type.hpp>
#endif

GPRAT_NS_BEGIN

#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name)                                                                       \
    static std::atomic<std::uint64_t> name(0);                                                                         \
    std::uint64_t get_##name(bool reset) { return hpx::util::get_and_reset_value(name, reset); }

GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_allocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_deallocations)

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR

void track_tile_data_allocation(std::size_t /*size*/) { tile_data_allocations += 1; }

void track_tile_data_deallocation(std::size_t /*size*/) { tile_data_deallocations += 1; }

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
// These are non-public functions of their respective CUs.
namespace detail
{
void register_fp32_performance_counters();
void register_fp64_performance_counters();
}  // namespace detail

void register_performance_counters()
{
    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name, stats_expr)                                                           \
    hpx::performance_counters::install_counter_type(                                                                   \
        name,                                                                                                          \
        [](bool reset) { return hpx::util::get_and_reset_value(stats_expr, reset); },                                  \
        #stats_expr,                                                                                                   \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_data/num_allocations", tile_data_allocations);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_data/num_deallocations", tile_data_deallocations);

#undef GPRAT_MAKE_STATISTICS_ACCESSOR

    detail::register_fp32_performance_counters();
    detail::register_fp64_performance_counters();
}
#else
void register_performance_counters()
{
    // no-op for binary compatibility
}
#endif

GPRAT_NS_END
