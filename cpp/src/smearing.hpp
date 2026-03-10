#pragma once

#include <algorithm>
#include <string>
#include "lattice.hpp"
#include "su2.hpp"

namespace smear {

inline int pbc(int x, int L) {
    int y = x % L;
    return (y < 0) ? y + L : y;
}

struct Site3D {
    int x, y, z;
};

inline Site3D shift_site(int x, int y, int z, int dir, int step, int L) {
    if (dir == 0) x = pbc(x + step, L);
    if (dir == 1) y = pbc(y + step, L);
    if (dir == 2) z = pbc(z + step, L);
    return {x, y, z};
}

/*
 * EDIT ONLY THESE HELPERS IF YOUR LATTICE API USES DIFFERENT NAMES.
 */
inline const SU2& get_link(const Lattice3D& lat, int x, int y, int z, int mu) {
    return lat.link(x, y, z, mu);
}

inline SU2& set_link(Lattice3D& lat, int x, int y, int z, int mu) {
    return lat.link(x, y, z, mu);
}

inline SU2 dag(const SU2& u) {
    return u.dagger();
}

inline SU2 linear_combo(const SU2& a, double ca, const SU2& b, double cb) {
    return SU2(
        ca * a.a0 + cb * b.a0,
        ca * a.a1 + cb * b.a1,
        ca * a.a2 + cb * b.a2,
        ca * a.a3 + cb * b.a3
    );
}

inline SU2 linear_combo4(
    const SU2& a, double ca,
    const SU2& b, double cb,
    const SU2& c, double cc,
    const SU2& d, double cd
) {
    return SU2(
        ca * a.a0 + cb * b.a0 + cc * c.a0 + cd * d.a0,
        ca * a.a1 + cb * b.a1 + cc * c.a1 + cd * d.a1,
        ca * a.a2 + cb * b.a2 + cc * c.a2 + cd * d.a2,
        ca * a.a3 + cb * b.a3 + cc * c.a3 + cd * d.a3
    );
}

inline SU2 project_to_su2(const SU2& x) {
    SU2 y = x;
    y.normalize();
    return y;
}

inline void copy_lattice(Lattice3D& dst, const Lattice3D& src, int L) {
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            for (int z = 0; z < L; ++z) {
                for (int mu = 0; mu < 3; ++mu) {
                    set_link(dst, x, y, z, mu) = get_link(src, x, y, z, mu);
                }
            }
        }
    }
}

/*
 * Spatial APE smearing in 3D SU(2):
 * - smear only spatial links mu = 0,1
 * - leave time-like links mu = 2 unchanged
 * - use only spatial staples, so each spatial link gets 2 staples
 */
inline void ape_smear_spatial_only(Lattice3D& lat, int L, int n_steps, double alpha) {
    Lattice3D current = lat;
    Lattice3D next = lat;

    for (int step = 0; step < n_steps; ++step) {
        copy_lattice(next, current, L);

        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < L; ++y) {
                for (int z = 0; z < L; ++z) {
                    for (int mu = 0; mu < 2; ++mu) {
                        const int nu = 1 - mu;  // only the other spatial direction

                        const Site3D x_plus_nu  = shift_site(x, y, z, nu, +1, L);
                        const Site3D x_plus_mu  = shift_site(x, y, z, mu, +1, L);
                        const Site3D x_minus_nu = shift_site(x, y, z, nu, -1, L);
                        const Site3D xmn_plus_mu = shift_site(x_minus_nu.x, x_minus_nu.y, x_minus_nu.z, mu, +1, L);

                        const SU2& U_mu_x = get_link(current, x, y, z, mu);
                        const SU2& U_nu_x = get_link(current, x, y, z, nu);

                        const SU2& U_mu_xpn = get_link(current, x_plus_nu.x, x_plus_nu.y, x_plus_nu.z, mu);
                        const SU2& U_nu_xpm = get_link(current, x_plus_mu.x, x_plus_mu.y, x_plus_mu.z, nu);

                        const SU2& U_nu_xmn = get_link(current, x_minus_nu.x, x_minus_nu.y, x_minus_nu.z, nu);
                        const SU2& U_mu_xmn = get_link(current, x_minus_nu.x, x_minus_nu.y, x_minus_nu.z, mu);
                        const SU2& U_nu_xmnpm = get_link(current, xmn_plus_mu.x, xmn_plus_mu.y, xmn_plus_mu.z, nu);

                        SU2 staple_fwd = U_nu_x * U_mu_xpn * dag(U_nu_xpm);
                        SU2 staple_bwd = dag(U_nu_xmn) * U_mu_xmn * U_nu_xmnpm;

                        SU2 staple_sum = linear_combo(staple_fwd, 1.0, staple_bwd, 1.0);

                        SU2 mixed = linear_combo(U_mu_x, 1.0 - alpha, staple_sum, 0.5 * alpha);
                        set_link(next, x, y, z, mu) = project_to_su2(mixed);
                    }

                    // keep time-like links unchanged
                    set_link(next, x, y, z, 2) = get_link(current, x, y, z, 2);
                }
            }
        }

        copy_lattice(current, next, L);
    }

    copy_lattice(lat, current, L);
}

} // namespace smear
