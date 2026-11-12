"""Minimal HepMC3 ASCII writer (readable by npsim/ddsim, the EIC afterburner and HepMC3 tools)."""

PDG_E, PDG_P, PDG_D = 11, 2212, 1000010020
PDG_X = 99  # generator-specific code; X is written with status 2 so it is not tracked


class HepMC3Writer:
    def __init__(self, path):
        self.f = open(path, "w")
        self.f.write("HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n")
        self.f.write("W weight\n")
        self.f.write("T tagdis-toymc\\|0.1\\|tagged DIS e+D->e'+X+p\n")

    @staticmethod
    def _p(i, mother, pid, v, m, status):
        return f"P {i} {mother} {pid} {v[1]:.10e} {v[2]:.10e} {v[3]:.10e} {v[0]:.10e} {m:.10e} {status}\n"

    def write(self, evno, weight, k, PD, kp, pp, X, masses, attrs=None):
        me, mD, mp, mX = masses
        lines = [f"E {evno} 1 5\n", "U GEV MM\n", f"W {weight:.10e}\n"]
        for name, val in (attrs or {}).items():
            lines.append(f"A 0 {name} {val}\n")
        lines += [
            self._p(1, 0, PDG_E, k, me, 4),
            self._p(2, 0, PDG_D, PD, mD, 4),
            "V -1 0 [1,2]\n",
            self._p(3, -1, PDG_E, kp, me, 1),
            self._p(4, -1, PDG_P, pp, mp, 1),
            self._p(5, -1, PDG_X, X, mX, 2),
        ]
        self.f.write("".join(lines))

    def close(self):
        self.f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
        self.f.close()
