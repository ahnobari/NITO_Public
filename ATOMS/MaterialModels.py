import numpy as np


class MaterialModel:
    def __init__(self, E=1.0, nu=0.33, n_sets=1):
        self.E = E
        self.nu = nu
        self.n_sets = n_sets

    def __call__(self, rho, iteration, **kwargs):
        return self.__forward__(rho, iteration, **kwargs)

    def evaluate_constraint(self, rho, dg, **kwargs):
        return self.__con__(rho, dg, **kwargs)

    def ocP(self, df, dg, rho, **kwargs):
        return self.__ocP__(df, dg, rho, **kwargs)

    def base_properties(self):
        return {"E": self.E, "nu": self.nu, "n_sets": self.n_sets}

    def grad(self, rho, iteration, **kwargs):
        return self.__backward__(rho, iteration, **kwargs)

    def is_terminal(self, iteration):
        return True

    def init_desvars(self, nel):
        return self.__initvars__(nel)


class SingleMaterial(MaterialModel):
    def __init__(
        self,
        E=1.0,
        nu=0.33,
        void=1e-6,
        penalty=3.0,
        volume_fraction=0.25,
        penalty_schedule=None,
    ):
        super().__init__(E=E, nu=nu, n_sets=1)
        self.void = void
        self.penalty = penalty
        self.volume_fraction = volume_fraction
        self.penalty_schedule = penalty_schedule

    def __forward__(self, rho, iteration, plain=False, np=np):
        if plain:
            rho = np.clip(rho, self.void, 1.0)
            return rho
        else:
            pen = self.penalty

            if self.penalty_schedule is not None:
                pen = self.penalty_schedule(self.penalty, iteration)

            rho = rho**pen
            rho = np.clip(rho, self.void, 1.0)

            return rho

    def __backward__(self, rho, iteration, np=np):

        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, iteration)

        rho = np.clip(rho, self.void, 1.0)

        df = pen * rho ** (pen - 1)

        return df

    def __con__(self, rho, dg, np=np):
        V = (rho.squeeze() * dg).sum()

        if V / dg.sum() <= self.volume_fraction:
            return np.array([True])
        else:
            return np.array([False])

    def __ocP__(self, df, dg, rho, np=np):

        ocP = rho * np.nan_to_num(np.sqrt(-df / dg.reshape(-1, 1)), nan=0)

        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3

        return ocP

    def __initvars__(self, nel):
        return np.ones([nel, 1])

    def is_terminal(self, iteration):
        if self.penalty_schedule is not None:
            return self.penalty_schedule(self.penalty, iteration) == self.penalty
        else:
            return True


class PenalizedMultiMaterial(MaterialModel):
    def __init__(
        self,
        E=1.0,
        nu=0.33,
        n_material=2,
        mass=np.array([[0.25], [0.25]]),
        E_mat=np.array([[1.0], [0.5]]),
        rho_mat=None,
        void=1e-6,
        penalty=3.0,
        penalty_schedule=None,
    ):
        super().__init__(E=E, nu=nu, n_sets=n_material)
        self.n_material = n_material
        self.mass = mass
        self.E_mat = E_mat
        self.rho_mat = rho_mat
        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule

        if self.E_mat.shape[0] != self.n_material:
            raise ValueError(
                "E_mat must have the same number of materials as n_material"
            )

        if self.rho_mat is not None:
            if self.rho_mat.shape[0] != self.n_material:
                raise ValueError(
                    "rho_mat must have the same number of materials as n_material"
                )

            if self.mass.shape[0] != 1:
                raise ValueError(
                    "if rho_mat is not None, mass must have only one value"
                )

            self.mass = self.mass.flatten()[0]
        else:
            if self.mass.shape[0] != self.n_material:
                raise ValueError(
                    "Mass and E_mat must have the same number of materials"
                )

    def __forward__(self, rho, iteration, plain=False, np=np):
        if plain:
            rho = rho @ np.array(self.E_mat)
            rho = np.clip(rho, self.void, 1.0)
            return rho
        else:
            pen = self.penalty

            if self.penalty_schedule is not None:
                pen = self.penalty_schedule(self.penalty, iteration)
            rho = np.clip(rho, self.void, 1.0)
            rho_ = rho**pen
            rho__ = 1 - rho_
            rho_ *= (
                rho__[
                    :,
                    np.where(~np.eye(self.n_material, dtype=bool))[1].reshape(
                        self.n_material, -1
                    ),
                ]
                .transpose(1, 0, 2)
                .prod(axis=-1)
                .T
            )

            E = rho_ @ np.array(self.E_mat)
            E = np.clip(E, self.void, np.inf)
            return E

    def __backward__(self, rho, iteration, np=np):

        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, iteration)

        rho = np.clip(rho, self.void, 1.0)
        rho_ = pen * rho ** (pen - 1)
        rho__ = 1 - rho**pen
        rho___ = rho**pen

        d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
        d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
        d = d.prod(axis=-1).transpose(0, 2, 1)

        mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
        mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

        d *= mul
        d = d @ np.array(self.E_mat)

        df_ = d.squeeze().T

        return df_

    def __con__(self, rho, dg, np=np):
        if self.rho_mat is None:
            V = (rho * dg.reshape(-1, 1)).sum(axis=0)

            return (V / dg.sum() - np.array(self.mass).reshape(-1)) <= 0

        else:
            V = ((rho * dg.reshape(-1, 1)) @ self.rho_mat).sum()

            return np.ones([self.n_material], dtype=bool) * (
                (V / dg.sum()) <= np.array(self.mass)
            )

    def __ocP__(self, df, dg, rho, np=np):

        # rho = np.clip(rho, self.void, np.inf)
        ocP = rho * np.nan_to_num(np.sqrt(-df / dg.reshape(-1, 1)), nan=0)

        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3

        return ocP

    def __initvars__(self, nel):
        rho = np.ones([nel, self.n_material])
        rho[:, 1:] = 0.0
        return rho

    def is_terminal(self, iteration):
        if self.penalty_schedule is not None:
            return self.penalty_schedule(self.penalty, iteration) == self.penalty
        else:
            return True

class PenalizedMultiMaterialRP(MaterialModel):
    def __init__(
        self,
        E=1.0,
        nu=0.33,
        n_material=2,
        mass=np.array([[0.25], [0.25]]),
        E_mat=np.array([[1.0], [0.5]]),
        rho_mat=None,
        void=1e-6,
        penalty=3.0,
        penalty_schedule=None,
    ):
        super().__init__(E=E, nu=nu, n_sets=n_material)
        self.n_material = n_material
        self.mass = mass
        self.E_mat = E_mat
        self.rho_mat = rho_mat
        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule

        if self.E_mat.shape[0] != self.n_material:
            raise ValueError(
                "E_mat must have the same number of materials as n_material"
            )

        if self.rho_mat is not None:
            if self.rho_mat.shape[0] != self.n_material:
                raise ValueError(
                    "rho_mat must have the same number of materials as n_material"
                )

            if self.mass.shape[0] != 1:
                raise ValueError(
                    "if rho_mat is not None, mass must have only one value"
                )

            self.mass = self.mass.flatten()[0]
        else:
            if self.mass.shape[0] != self.n_material:
                raise ValueError(
                    "Mass and E_mat must have the same number of materials"
                )

    def __forward__(self, rho, iteration, plain=False, np=np):
        if plain:
            rho = rho @ np.array(self.E_mat)
            rho = np.clip(rho, self.void, 1.0)
            return rho
        else:
            pen = self.penalty

            if self.penalty_schedule is not None:
                pen = self.penalty_schedule(self.penalty, iteration)
            rho = np.clip(rho, self.void, 1.0)
            rho_ = rho**pen
            rho__ = (1 - rho)**pen
            rho_ *= (
                rho__[
                    :,
                    np.where(~np.eye(self.n_material, dtype=bool))[1].reshape(
                        self.n_material, -1
                    ),
                ]
                .transpose(1, 0, 2)
                .prod(axis=-1)
                .T
            )

            E = rho_ @ np.array(self.E_mat)
            E = np.clip(E, self.void, np.inf)
            return E

    def __backward__(self, rho, iteration, np=np):

        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, iteration)

        rho = np.clip(rho, self.void, 1.0)
        rho_ = pen * rho ** (pen - 1)
        rho_h = pen * (1 - rho) ** (pen - 1)
        rho__ = (1 - rho)**pen
        rho___ = rho**pen

        d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
        d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
        d = d.prod(axis=-1).transpose(0, 2, 1)

        mul = -rho_h.T[:, :, np.newaxis].repeat(self.n_material, -1)
        mul[np.arange(self.n_material), :, np.arange(self.n_material)] = rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)[np.arange(self.n_material), :, np.arange(self.n_material)]

        d *= mul
        d = d @ np.array(self.E_mat)

        df_ = d.squeeze().T

        return df_

    def __con__(self, rho, dg, np=np):
        if self.rho_mat is None:
            V = (rho * dg.reshape(-1, 1)).sum(axis=0)

            return (V / dg.sum() - np.array(self.mass).reshape(-1)) <= 0

        else:
            V = ((rho * dg.reshape(-1, 1)) @ self.rho_mat).sum()

            return np.ones([self.n_material], dtype=bool) * (
                (V / dg.sum()) <= np.array(self.mass)
            )

    def __ocP__(self, df, dg, rho, np=np):

        # rho = np.clip(rho, self.void, np.inf)
        ocP = rho * np.nan_to_num(np.sqrt(-df / dg.reshape(-1, 1)), nan=0)

        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3

        return ocP

    def __initvars__(self, nel):
        return np.ones([nel, self.n_material]) * 0.5

    def is_terminal(self, iteration):
        if self.penalty_schedule is not None:
            return self.penalty_schedule(self.penalty, iteration) == self.penalty
        else:
            return True