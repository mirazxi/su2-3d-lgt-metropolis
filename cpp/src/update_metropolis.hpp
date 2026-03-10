#pragma once

#include <cmath>
#include <cstddef>
#include "lattice.hpp"
#include "rng.hpp"

namespace metro {

inline void shift_site(
    const Lattice3D& lat,
    int x, int y, int z,
    int mu, int dir,
    int& xs, int& ys, int& zs
) {
    xs = x;
    ys = y;
    zs = z;

    if (mu == 0) {
        xs = lat.shift_x(x, dir);
    } else if (mu == 1) {
        ys = lat.shift_y(y, dir);
    } else if (mu == 2) {
        zs = lat.shift_z(z, dir);
    }
}

inline SU2 forward_plaquette(
    const Lattice3D& lat,
    int x, int y, int z,
    int mu, int nu
) {
    int x_mu, y_mu, z_mu;
    int x_nu, y_nu, z_nu;

    shift_site(lat, x, y, z, mu, +1, x_mu, y_mu, z_mu);
    shift_site(lat, x, y, z, nu, +1, x_nu, y_nu, z_nu);

    const SU2& U_mu_x     = lat.link(x,    y,    z,    mu);
    const SU2& U_nu_xpmu  = lat.link(x_mu, y_mu, z_mu, nu);
    const SU2& U_mu_xpnu  = lat.link(x_nu, y_nu, z_nu, mu);
    const SU2& U_nu_x     = lat.link(x,    y,    z,    nu);

    return U_mu_x * U_nu_xpmu * U_mu_xpnu.dagger() * U_nu_x.dagger();
}

inline SU2 backward_plaquette(
    const Lattice3D& lat,
    int x, int y, int z,
    int mu, int nu
) {
    int x_mnu, y_mnu, z_mnu;
    int x_mnu_pmu, y_mnu_pmu, z_mnu_pmu;

    shift_site(lat, x, y, z, nu, -1, x_mnu, y_mnu, z_mnu);
    shift_site(lat, x_mnu, y_mnu, z_mnu, mu, +1, x_mnu_pmu, y_mnu_pmu, z_mnu_pmu);

    const SU2& U_nu_xmnu      = lat.link(x_mnu,      y_mnu,      z_mnu,      nu);
    const SU2& U_mu_x         = lat.link(x,          y,          z,          mu);
    const SU2& U_nu_xmnu_pmu  = lat.link(x_mnu_pmu,  y_mnu_pmu,  z_mnu_pmu,  nu);
    const SU2& U_mu_xmnu      = lat.link(x_mnu,      y_mnu,      z_mnu,      mu);

    return U_nu_xmnu * U_mu_x * U_nu_xmnu_pmu.dagger() * U_mu_xmnu.dagger();
}

inline double local_action(
    const Lattice3D& lat,
    int x, int y, int z,
    int mu,
    double beta
) {
    double sum = 0.0;

    for (int nu = 0; nu < 3; ++nu) {
        if (nu == mu) {
            continue;
        }

        SU2 pf = forward_plaquette(lat, x, y, z, mu, nu);
        SU2 pb = backward_plaquette(lat, x, y, z, mu, nu);

        sum += 1.0 - 0.5 * pf.trace();
        sum += 1.0 - 0.5 * pb.trace();
    }

    return beta * sum;
}

inline double metropolis_sweep(
    Lattice3D& lat,
    double beta,
    double eps,
    RNG& rng
) {
    std::size_t accepted = 0;
    std::size_t total = 0;

    for (int z = 0; z < lat.Lz(); ++z) {
        for (int y = 0; y < lat.Ly(); ++y) {
            for (int x = 0; x < lat.Lx(); ++x) {
                for (int mu = 0; mu < 3; ++mu) {
                    SU2 old_link = lat.link(x, y, z, mu);
                    double old_s = local_action(lat, x, y, z, mu, beta);

                    SU2 proposal = rng.small_su2(eps) * old_link;
                    lat.link(x, y, z, mu) = proposal;

                    double new_s = local_action(lat, x, y, z, mu, beta);
                    double dS = new_s - old_s;

                    bool accept = (dS <= 0.0) || (rng.uniform01() < std::exp(-dS));

                    if (accept) {
                        ++accepted;
                    } else {
                        lat.link(x, y, z, mu) = old_link;
                    }

                    ++total;
                }
            }
        }
    }

    return static_cast<double>(accepted) / static_cast<double>(total);
}

} // namespace metro
