#pragma once

#include <cstdint>
#include <random>
#include "su2.hpp"

class RNG {
public:
    explicit RNG(std::uint64_t seed = 123456789ull)
        : gen_(seed), uni_(0.0, 1.0), gauss_(0.0, 1.0) {}

    double uniform01() {
        return uni_(gen_);
    }

    double normal() {
        return gauss_(gen_);
    }

    SU2 small_su2(double eps) {
        SU2 r(1.0, eps * normal(), eps * normal(), eps * normal());
        r.normalize();

        if (r.a0 < 0.0) {
            r.a0 = -r.a0;
            r.a1 = -r.a1;
            r.a2 = -r.a2;
            r.a3 = -r.a3;
        }

        return r;
    }

    SU2 random_su2() {
        SU2 r(normal(), normal(), normal(), normal());
        r.normalize();

        if (r.a0 < 0.0) {
            r.a0 = -r.a0;
            r.a1 = -r.a1;
            r.a2 = -r.a2;
            r.a3 = -r.a3;
        }

        return r;
    }

private:
    std::mt19937_64 gen_;
    std::uniform_real_distribution<double> uni_;
    std::normal_distribution<double> gauss_;
};
