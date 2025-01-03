#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <thread>
#include <limits>

#include <Eigen/Dense>

#include <ncurses.h>
#include <matplot/matplot.h>

struct Link
{
    double L; // Length
    double m; // Mass
};

template <int N, int M>
struct State
{
    Eigen::Matrix<double, N, 1> q = Eigen::Matrix<double, N, 1>::Zero(); // Lagrange generalized coordinates (angles)
    Eigen::Matrix<double, N, 1> qdot = Eigen::Matrix<double, N, 1>::Zero();
    Eigen::Matrix<double, N, M> dq_dT = Eigen::Matrix<double, N, M>::Zero();
    Eigen::Matrix<double, N, M> dqdot_dT = Eigen::Matrix<double, N, M>::Zero();
};

template <int M>
struct Cost
{
    double c = 0; // Value of cost function
    Eigen::Matrix<double, 1, M> dc_dT = Eigen::Matrix<double, 1, M>::Zero();
};

template <int M>
struct LineSearchResult
{
    double alpha;
    Eigen::Matrix<double, M, 1> x;
    Eigen::Matrix<double, M, 1> s;
    Cost<M> cost;
};

template <int N, int M>
std::array<State<N, M>, M> simulate(const std::array<Link, N> &link,
                                    const Eigen::Matrix<double, N, 1> &q0,
                                    const Eigen::Matrix<double, N, 1> &qdot0,
                                    const Eigen::Matrix<double, M, 1> &T,
                                    double g,
                                    double dt)
{
    std::array<State<N, M>, M> state;

    state[0].q = q0;
    state[0].qdot = qdot0;

    std::array<Eigen::Matrix<double, N, N>, N> dA_dq, dA_dqdot;
    std::array<Eigen::Matrix<double, N, 1>, N> dB_dq, dB_dqdot;
    for (int m = 0; m < M - 1; ++m)
    {
        Eigen::Matrix<double, N, N> A = Eigen::Matrix<double, N, N>::Zero();
        Eigen::Matrix<double, N, 1> B = Eigen::Matrix<double, N, 1>::Zero();

        Eigen::Matrix<double, N, 1> dB_dT = Eigen::Matrix<double, N, 1>::Zero();
        for (size_t n = 0; n < N; ++n)
        {
            dA_dq[n].setZero();
            dA_dqdot[n].setZero();
            dB_dq[n].setZero();
            dB_dqdot[n].setZero();
        }

        B(0) += T(m);
        dB_dT(0) += 1;

        for (size_t i = 0; i < N; ++i)
        {
            const double c_qi = std::cos(state[m].q(i));
            const double s_qi = std::sin(state[m].q(i));

            for (size_t n = 0; n < N; ++n)
            {
                const double Li_mn = link[i].L * link[n].m;
                const double Li_mn_g = Li_mn * g;

                B(i) += -Li_mn_g * s_qi;
                dB_dq[i](i) += -Li_mn_g * c_qi;

                if (i <= n)
                {
                    for (size_t j = 0; j <= n; ++j)
                    {
                        const double dqij = state[m].q(i) - state[m].q(j);
                        const double c_dqij = std::cos(dqij);
                        const double s_dqij = std::sin(dqij);
                        const double Li_Lj_mn = Li_mn * link[j].L;
                        const double Li_Lj_mn_qdotj = Li_Lj_mn * state[m].qdot(j);
                        const double Li_Lj_mn_qdotj2 = Li_Lj_mn_qdotj * state[m].qdot(j);

                        B(i) += -Li_Lj_mn_qdotj2 * s_dqij;
                        dB_dq[i](i) += -Li_Lj_mn_qdotj2 * c_dqij;
                        dB_dq[j](i) += +Li_Lj_mn_qdotj2 * c_dqij;
                        dB_dqdot[j](i) += -2 * Li_Lj_mn_qdotj * s_dqij;

                        A(i, j) += Li_Lj_mn * c_dqij;
                        dA_dq[i](i, j) += -Li_Lj_mn * s_dqij;
                        dA_dq[j](i, j) += +Li_Lj_mn * s_dqij;
                    }
                }
            }
        }

        const Eigen::FullPivLU<Eigen::Matrix<double, N, N>> lu_A{A};
        const Eigen::Matrix<double, N, 1> qddot = lu_A.solve(B);

        Eigen::Matrix<double, N, M> inter = Eigen::Matrix<double, N, M>::Zero();
        inter.col(m) += dB_dT;
        for (int i = 0; i < N; ++i)
        {
            inter += (-dA_dq[i] * qddot + dB_dq[i]) * state[m].dq_dT.row(i);
            inter += (-dA_dqdot[i] * qddot + dB_dqdot[i]) * state[m].dqdot_dT.row(i);
        }
        const Eigen::Matrix<double, N, M> dqddot_dT = lu_A.solve(inter);

        state[m + 1].q = state[m].q + dt * state[m].qdot;
        state[m + 1].dq_dT = state[m].dq_dT + dt * state[m].dqdot_dT;

        state[m + 1].qdot = state[m].qdot + dt * qddot;
        state[m + 1].dqdot_dT = state[m].dqdot_dT + dt * dqddot_dT;
    }

    return state;
}

template <int N, int M>
Cost<M> compute_cost(const std::array<State<N, M>, M> &state,
                     const Eigen::Matrix<double, M, 1> &T,
                     const Eigen::Matrix<double, N, 1> &q_ref,
                     double Q_q,
                     double Q_qdot,
                     double Q_T)
{
    Cost<M> cost;
    for (int m = 0; m < M; ++m)
    {
        const Eigen::Matrix<double, N, 1> dq = state[m].q - q_ref;
        cost.c += Q_q * dq.squaredNorm() / 2;
        cost.dc_dT += Q_q * dq.transpose() * state[m].dq_dT;

        cost.c += Q_qdot * state[m].qdot.squaredNorm() / 2;
        cost.dc_dT += Q_qdot * state[m].qdot.transpose() * state[m].dqdot_dT;

        cost.c += Q_T * T(m) * T(m) / 2;
        cost.dc_dT(m) += Q_T * T(m);
    }
    return cost;
}

template <int M, typename CostFunction>
LineSearchResult<M> line_search(const CostFunction &f,
                                const Eigen::Matrix<double, M, 1> &x,
                                const Eigen::Matrix<double, M, 1> &p,
                                const Cost<M> &cost)
{
    const double c1 = 1e-4, c2 = 0.9;

    LineSearchResult<M> result;
    result.alpha = 1;
    while (true)
    {
        result.s = result.alpha * p;
        result.x = x + result.s;
        result.cost = f(result.x);

        if (result.alpha == 0)
        {
            break;
        }

        if (result.cost.c <= cost.c)
        {
            // Wolfe conditions
            const double dc_dT_p = cost.dc_dT * p;
            if ((result.cost.c <= cost.c + c1 * result.alpha * dc_dT_p) || (-result.cost.dc_dT * p <= -c2 * dc_dT_p))
            {
                break;
            }
        }

        result.alpha /= 2;
    }

    return result;
}

template <int M, typename CostFunction>
Eigen::Matrix<double, M, 1> bfgs(const CostFunction &f, const Eigen::Matrix<double, M, 1> &x0, double tol, int max_iter)
{
    Eigen::Matrix<double, M, 1> x = x0;

    Cost<M> cost = f(x);
    Eigen::Matrix<double, M, M> H = Eigen::Matrix<double, M, M>::Identity();
    for (int i = 1;; ++i)
    {
        const Eigen::Matrix<double, M, 1> p = -H * cost.dc_dT.transpose();
        const LineSearchResult<M> lsr = line_search(f, x, p, cost);
        const Eigen::Matrix<double, M, 1> y = (lsr.cost.dc_dT - cost.dc_dT).transpose();

        const double sT_y = lsr.s.transpose() * y;
        const Eigen::Matrix<double, M, M> s_yT = lsr.s * y.transpose();
        H += (sT_y + y.transpose() * H * y) * (lsr.s * lsr.s.transpose()) / (sT_y * sT_y) - (H * s_yT.transpose() + s_yT * H) / sT_y;
        cost = lsr.cost;
        x = lsr.x;

        if (i >= max_iter)
        {
            std::cerr << "\rbfgs: Maximum number of iterations reached, returning incomplete solution\r\n";
            break;
        }
        if (cost.dc_dT.norm() < tol)
        {
            break;
        }
    }

    return x;
}

int main()
{
    const double g = 1;     // Gravitational acceleration
    const double dt = 0.05; // Time step

    const std::array link{Link{.L = 1, .m = 1}, Link{.L = 1, .m = 1}, Link{.L = 1, .m = 1}}; // Pendulum link lengths and masses

    const int M = 100; // Receeding horizon, number of time steps
    constexpr int N = link.size();

    Eigen::Matrix<double, N, 1> q = Eigen::Matrix<double, N, 1>::Constant(0.0);     // Initial angles
    Eigen::Matrix<double, N, 1> qdot = Eigen::Matrix<double, N, 1>::Constant(0.0);  // Initial angular velocities
    Eigen::Matrix<double, N, 1> q_ref = Eigen::Matrix<double, N, 1>::Constant(0.0); // Initial reference angles
    Eigen::Matrix<double, M, 1> T = Eigen::Matrix<double, M, 1>::Constant(0.0);

    const auto cost_function = [&link, &q, &qdot, &q_ref, &g, &dt](const Eigen::Matrix<double, M, 1> &T) -> Cost<M>
    {
        const double Q_q = 1.0;
        const double Q_qdot = 0.1;
        const double Q_T = 0.1;
        return compute_cost<N, M>(simulate<N, M>(link, q, qdot, T, g, dt), T, q_ref, Q_q, Q_qdot, Q_T);
    };

    const double tol = 1e-3;
    const int max_iter = 100;

    double total_L = 0.0;
    for (int n = 0; n < N; ++n)
    {
        total_L += link[n].L;
    }
    const double axis_lim = 1.2 * total_L;

    auto fig = matplot::figure(true);
    fig->size(400, 400);
    auto ax = fig->add_axes();
    ax->xlim({-axis_lim, axis_lim});
    ax->ylim({-axis_lim, axis_lim});
    ax->axis(matplot::equal);
    auto line = ax->plot(std::vector<double>{0.0}, std::vector<double>{0.0}, "k-o");

    initscr();
    cbreak();
    nodelay(stdscr, true);
    keypad(stdscr, true);
    noecho();

    while (true)
    {
        const auto time_start = std::chrono::high_resolution_clock::now();

        std::vector<double> x_data = {0.0}, y_data = {0.0};
        for (int n = 0; n < N; ++n)
        {
            x_data.push_back(x_data.back() - link[n].L * std::sin(q(n)));
            y_data.push_back(y_data.back() - link[n].L * std::cos(q(n)));
        }
        line->x_data(x_data);
        line->y_data(y_data);
        fig->draw();

        const int ch = getch();
        if (ch == 'q')
        {
            break;
        }
        else
        {
            const int n = ch - 49;
            if (n >= 0 && n < q_ref.size())
            {
                q_ref(n) = M_PI - q_ref(n);
            }
        }

        T = bfgs<M>(cost_function, T, tol, max_iter);
        std::array<State<N, 2>, 2> state = simulate<N, 2>(link, q, qdot, T(Eigen::seq(0, 1)), g, dt);

        q = state[1].q;
        qdot = state[1].qdot;
        T(Eigen::seqN(0, M - 1)) = T(Eigen::seqN(1, M - 1));

        const auto time_current = std::chrono::high_resolution_clock::now();
        const double elapsed_time = std::chrono::duration<double>(time_current - time_start).count();
        if (elapsed_time < dt)
        {
            std::this_thread::sleep_for(std::chrono::duration<double>(dt - elapsed_time));
        }
    }

    endwin();

    return 0;
}
