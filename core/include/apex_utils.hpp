#ifndef APEX_STEPS_H
#define APEX_STEPS_H

#if GPRAT_APEX_STEPS
#include <apex_api.hpp>
#endif

#include <hpx/future.hpp>

/// @brief Alias for obtaining the current high-resolution time point.
inline auto now = std::chrono::high_resolution_clock::now;

/// @brief Computes the duration in nanoseconds between the current time and a given start time.
inline double diff(const std::chrono::high_resolution_clock::time_point &start_time)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start_time).count());
}

/**
 * @brief Initializes a new apex timer newTimer for the current scope.
 *
 * @param newTimer Identifier of the new timer variable to be declared
 */
#define GPRAT_START_TIMER(newTimer) auto newTimer = now()

/**
 * @brief Blocks execution until all provided HPX futures are ready and samples the duration of APEX timer oldTimer with
 * label oldLabel.
 *
 * @param oldTimer Identifier of the existing timer variable to be sampled
 * @param oldLabel String label associated with the measured duration
 * @param ...      Variadic arguments representing HPX futures to wait on
 */
#define GPRAT_STOP_TIMER(oldTimer, oldLabel, ...)                                                                      \
    hpx::wait_all(__VA_ARGS__);                                                                                        \
    apex::sample_value(oldLabel, diff(oldTimer))

// Macros GPRAT_START_STEP and GPRAT_END_STEP are conditionally defined based on the value of GPRAT_APEX_STEPS. They are
// identical to GPRAT_START_TIMER and GPRAT_STOP_TIMER when GPRAT_APEX_STEPS=ON, otherwise they are defined as empty.
#if GPRAT_APEX_STEPS

/// @see GPRAT_START_TIMER
#define GPRAT_START_STEP(newTimer) GPRAT_START_TIMER(newTimer)

/// @see GPRAT_STOP_TIMER
#define GPRAT_END_STEP(oldTimer, oldLabel, ...) GPRAT_STOP_TIMER(oldTimer, oldLabel, __VA_ARGS__)

#else

// Empty macro definitions when GPRAT_APEX_STEPS=OFF
#define GPRAT_START_STEP(...)
#define GPRAT_END_STEP(...)

#endif  // GPRAT_APEX_STEPS

// NOTE: We could also create similar macros, e.g. for GPRAT_APEX_CHOLESKY. However, since GPRAT_APEX_CHOLESKY is only
// used in once, this would unnecessarily bloat this header file.

#endif  // APEX_STEPS_H
