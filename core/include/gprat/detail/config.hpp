#ifndef GPRAT_DETAIL_CONFIG_HPP
#define GPRAT_DETAIL_CONFIG_HPP

#pragma once

// clang-format off
#define GPRAT_NS gprat::v1
#define GPRAT_NS_BEGIN namespace gprat { inline namespace v1 {
#define GPRAT_NS_END } }
// clang-format on

#if defined(_MSC_VER) || defined(__BORLANDC__) || defined(__CODEGEARC__)
#if defined(GPRAT_DYN_LINK)
#if defined(GPRAT_SOURCE)
#define GPRAT_DECL __declspec(dllexport)
#else
#define GPRAT_DECL __declspec(dllimport)
#endif
#endif
#endif

#if !defined(GPRAT_DECL)
#define GPRAT_DECL
#endif

#endif
