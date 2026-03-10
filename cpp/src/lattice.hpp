#pragma once

#include <vector>
#include <stdexcept>
#include "su2.hpp"

class Lattice3D {
public:
    Lattice3D(int lx, int ly, int lz)
        : Lx_(lx), Ly_(ly), Lz_(lz), links_(static_cast<size_t>(lx) * ly * lz * 3, SU2::identity()) {
        if (lx <= 0 || ly <= 0 || lz <= 0) {
            throw std::invalid_argument("Lattice sizes must be positive.");
        }
    }

    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    int Lz() const { return Lz_; }
    int volume() const { return Lx_ * Ly_ * Lz_; }

    void cold_start() {
        for (auto& u : links_) {
            u = SU2::identity();
        }
    }

    SU2& link(int x, int y, int z, int mu) {
        return links_.at(index(x, y, z, mu));
    }

    const SU2& link(int x, int y, int z, int mu) const {
        return links_.at(index(x, y, z, mu));
    }

    int shift_x(int x, int dir) const { return mod(x + dir, Lx_); }
    int shift_y(int y, int dir) const { return mod(y + dir, Ly_); }
    int shift_z(int z, int dir) const { return mod(z + dir, Lz_); }

private:
    int Lx_, Ly_, Lz_;
    std::vector<SU2> links_;

    static int mod(int a, int n) {
        int r = a % n;
        return (r < 0) ? r + n : r;
    }

    size_t index(int x, int y, int z, int mu) const {
        x = mod(x, Lx_);
        y = mod(y, Ly_);
        z = mod(z, Lz_);
        if (mu < 0 || mu >= 3) {
            throw std::out_of_range("Direction mu must be 0, 1, or 2.");
        }
        return static_cast<size_t>((((z * Ly_) + y) * Lx_ + x) * 3 + mu);
    }
};
