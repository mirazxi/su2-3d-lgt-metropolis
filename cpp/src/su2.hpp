#pragma once

#include <cmath>
#include <ostream>

struct SU2 {
    double a0, a1, a2, a3;

    SU2() : a0(1.0), a1(0.0), a2(0.0), a3(0.0) {}
    SU2(double x0, double x1, double x2, double x3)
        : a0(x0), a1(x1), a2(x2), a3(x3) {}

    static SU2 identity() {
        return SU2(1.0, 0.0, 0.0, 0.0);
    }

    double norm2() const {
        return a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    }

    void normalize() {
        double n = std::sqrt(norm2());
        if (n > 0.0) {
            a0 /= n;
            a1 /= n;
            a2 /= n;
            a3 /= n;
        }
    }

    SU2 dagger() const {
        return SU2(a0, -a1, -a2, -a3);
    }

    double trace() const {
        return 2.0 * a0;
    }

    SU2 operator*(const SU2& other) const {
        return SU2(
            a0 * other.a0 - a1 * other.a1 - a2 * other.a2 - a3 * other.a3,
            a0 * other.a1 + a1 * other.a0 + a2 * other.a3 - a3 * other.a2,
            a0 * other.a2 - a1 * other.a3 + a2 * other.a0 + a3 * other.a1,
            a0 * other.a3 + a1 * other.a2 - a2 * other.a1 + a3 * other.a0
        );
    }
};

inline std::ostream& operator<<(std::ostream& os, const SU2& u) {
    os << "(" << u.a0 << ", " << u.a1 << ", " << u.a2 << ", " << u.a3 << ")";
    return os;
}
