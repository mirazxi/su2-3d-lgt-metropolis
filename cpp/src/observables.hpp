#pragma once

#include "lattice.hpp"

namespace obs {

inline void shift_site(
    const Lattice3D& lat,
    int x, int y, int z, int mu, int dir,
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

inline void neighbor(
    const Lattice3D& lat,
    int x, int y, int z, int mu,
    int& xp, int& yp, int& zp
) {
    shift_site(lat, x, y, z, mu, +1, xp, yp, zp);
}

inline SU2 plaquette_matrix(const Lattice3D& lat, int x, int y, int z, int mu, int nu) {
    int x_mu, y_mu, z_mu;
    int x_nu, y_nu, z_nu;

    neighbor(lat, x, y, z, mu, x_mu, y_mu, z_mu);
    neighbor(lat, x, y, z, nu, x_nu, y_nu, z_nu);

    const SU2& U_mu_x     = lat.link(x,    y,    z,    mu);
    const SU2& U_nu_xpmu  = lat.link(x_mu, y_mu, z_mu, nu);
    const SU2& U_mu_xpnu  = lat.link(x_nu, y_nu, z_nu, mu);
    const SU2& U_nu_x     = lat.link(x,    y,    z,    nu);

    return U_mu_x * U_nu_xpmu * U_mu_xpnu.dagger() * U_nu_x.dagger();
}

inline double average_plaquette(const Lattice3D& lat) {
    double sum = 0.0;

    for (int z = 0; z < lat.Lz(); ++z) {
        for (int y = 0; y < lat.Ly(); ++y) {
            for (int x = 0; x < lat.Lx(); ++x) {
                for (int mu = 0; mu < 3; ++mu) {
                    for (int nu = mu + 1; nu < 3; ++nu) {
                        SU2 up = plaquette_matrix(lat, x, y, z, mu, nu);
                        sum += 0.5 * up.trace();
                    }
                }
            }
        }
    }

    return sum / (3.0 * lat.volume());
}

inline SU2 rectangular_loop_at(
    const Lattice3D& lat,
    int x, int y, int z,
    int spatial_dir,
    int time_dir,
    int R,
    int T
) {
    SU2 loop = SU2::identity();
    int cx = x, cy = y, cz = z;

    for (int i = 0; i < R; ++i) {
        loop = loop * lat.link(cx, cy, cz, spatial_dir);
        int nx, ny, nz;
        shift_site(lat, cx, cy, cz, spatial_dir, +1, nx, ny, nz);
        cx = nx; cy = ny; cz = nz;
    }

    for (int i = 0; i < T; ++i) {
        loop = loop * lat.link(cx, cy, cz, time_dir);
        int nx, ny, nz;
        shift_site(lat, cx, cy, cz, time_dir, +1, nx, ny, nz);
        cx = nx; cy = ny; cz = nz;
    }

    for (int i = 0; i < R; ++i) {
        int px, py, pz;
        shift_site(lat, cx, cy, cz, spatial_dir, -1, px, py, pz);
        loop = loop * lat.link(px, py, pz, spatial_dir).dagger();
        cx = px; cy = py; cz = pz;
    }

    for (int i = 0; i < T; ++i) {
        int px, py, pz;
        shift_site(lat, cx, cy, cz, time_dir, -1, px, py, pz);
        loop = loop * lat.link(px, py, pz, time_dir).dagger();
        cx = px; cy = py; cz = pz;
    }

    return loop;
}

inline double average_wilson_loop_spacetime(
    const Lattice3D& lat,
    int R,
    int T,
    int time_dir = 2
) {
    double sum = 0.0;
    int count = 0;

    for (int spatial_dir = 0; spatial_dir < 3; ++spatial_dir) {
        if (spatial_dir == time_dir) {
            continue;
        }

        for (int z = 0; z < lat.Lz(); ++z) {
            for (int y = 0; y < lat.Ly(); ++y) {
                for (int x = 0; x < lat.Lx(); ++x) {
                    SU2 w = rectangular_loop_at(lat, x, y, z, spatial_dir, time_dir, R, T);
                    sum += 0.5 * w.trace();
                    ++count;
                }
            }
        }
    }

    return sum / static_cast<double>(count);
}

} // namespace obs
