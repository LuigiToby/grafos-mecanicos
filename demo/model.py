"""Modelo gráfico y planar del subconjunto EEZYbotARM usado en la demostración.

Las aristas representan articulaciones revolutas entre eslabones. Los cierres
vectoriales se generan desde una base fundamental del grafo. Las dimensiones,
masas y aproximaciones de inercia reproducen el cuaderno previo del proyecto.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, sin, pi
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


GRAVITY = 9.81
BODY_ORDER = ("J", "K", "L", "M", "N", "O", "P", "Q")
BODY_INDEX = {name: i for i, name in enumerate(BODY_ORDER)}
VERTICES = ("T",) + BODY_ORDER

# Geometría tomada del modelo previo, en metros.
LENGTHS = {
    "l1": 0.035,
    "l2": 0.080,
    "l3": 0.080,
    "l4": 0.035,
    "l5": 0.020,
    "l6": 0.020,
    "l7": 0.040,
    "l9": 0.01723,
    "l10": 0.0361,
}

# Orden y orientación de aristas usados para asignar el signo de las reacciones.
EDGES = (
    ("T", "J"), ("J", "P"), ("J", "N"), ("T", "K"), ("K", "L"),
    ("L", "P"), ("T", "M"), ("M", "N"), ("N", "O"), ("O", "Q"),
    ("Q", "P"),
)
EDGE_INDEX = {frozenset(edge): i for i, edge in enumerate(EDGES)}
TREE_EDGES = (
    ("T", "J"), ("J", "P"), ("J", "N"), ("T", "K"),
    ("K", "L"), ("T", "M"), ("N", "O"), ("O", "Q"),
)
CHORDS = (("L", "P"), ("M", "N"), ("Q", "P"))
TREE_PARENT_CHILD = (
    ("T", "J"), ("J", "P"), ("J", "N"), ("T", "K"),
    ("K", "L"), ("T", "M"), ("N", "O"), ("O", "Q"),
)

LENGTH = LENGTHS
MASS = {
    "J": 0.0141, "K": 0.0029, "L": 0.0029, "M": 0.0029,
    "N": 0.0035, "O": 0.0029, "P": 0.0096, "Q": 0.0062,
}


def edge_key(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _vec(x: float, y: float) -> np.ndarray:
    return np.array((x, y), dtype=float)


def _unit(theta: float, length: float = 1.0) -> np.ndarray:
    return length * _vec(cos(theta), sin(theta))


def rotation(theta: float) -> np.ndarray:
    c, s = cos(theta), sin(theta)
    return np.array(((c, -s), (s, c)), dtype=float)


def quarter_turn(v: np.ndarray) -> np.ndarray:
    return np.array((-v[1], v[0]), dtype=float)


# Coordenadas locales de cada pivote sobre cada eslabón. En T se fijan los
# pivotes del bastidor. J, K, L, M, O y Q son barras; N y P son ternarios.
PIN: dict[str, dict[frozenset[str], np.ndarray]] = {
    name: {} for name in VERTICES
}


def _pin(body: str, a: str, b: str, xy: Iterable[float]) -> None:
    PIN[body][edge_key(a, b)] = np.asarray(tuple(xy), dtype=float)


_pin("T", "T", "J", (0.0, 0.0))
_pin("J", "T", "J", (0.0, 0.0))
_pin("J", "J", "P", (LENGTH["l2"], 0.0))
_pin("J", "J", "N", (LENGTH["l2"], 0.0))
_pin("P", "J", "P", (0.0, 0.0))
_pin("N", "J", "N", (0.0, 0.0))

_pin("T", "T", "K", (0.0, 0.0))
_pin("K", "T", "K", (0.0, 0.0))
_pin("K", "K", "L", (LENGTH["l1"], 0.0))
_pin("L", "K", "L", (0.0, 0.0))
_pin("L", "L", "P", (LENGTH["l3"], 0.0))
_pin("P", "L", "P", (LENGTH["l4"], 0.0))

_pin("T", "T", "M", (LENGTH["l10"], LENGTH["l9"]))
_pin("M", "T", "M", (0.0, 0.0))
_pin("M", "M", "N", (LENGTH["l3"], 0.0))
_pin("N", "M", "N", (LENGTH["l7"], 0.0))
_pin("N", "N", "O", _unit(95.0 * pi / 180.0, LENGTH["l6"]))
_pin("O", "N", "O", (LENGTH["l3"], 0.0))
_pin("O", "O", "Q", (0.0, 0.0))
_pin("Q", "O", "Q", (LENGTH["l5"], 0.0))
_pin("Q", "Q", "P", (0.0, 0.0))
_pin("P", "Q", "P", (-LENGTH["l3"], 0.0))

COM_LOCAL = {
    "J": _vec(LENGTH["l2"] / 2, 0),
    "K": _vec(LENGTH["l1"] / 2, 0),
    "L": _vec(LENGTH["l3"] / 2, 0),
    "M": _vec(LENGTH["l3"] / 2, 0),
    "N": _vec(
        (LENGTH["l7"] + LENGTH["l6"] * cos(95.0 * pi / 180.0)) / 3,
        LENGTH["l6"] * sin(95.0 * pi / 180.0) / 3,
    ),
    "O": _vec(LENGTH["l3"] / 2, 0),
    "P": _vec((LENGTH["l4"] - LENGTH["l3"]) / 2, 0),
    "Q": _vec(LENGTH["l5"] / 2, 0),
}

INERTIA = {
    "J": MASS["J"] * LENGTH["l2"] ** 2 / 12,
    "K": MASS["K"] * LENGTH["l1"] ** 2 / 12,
    "L": MASS["L"] * LENGTH["l3"] ** 2 / 12,
    "M": MASS["M"] * LENGTH["l3"] ** 2 / 12,
    "N": MASS["N"] * (LENGTH["l6"] ** 2 + LENGTH["l7"] ** 2) / 18,
    "O": MASS["O"] * LENGTH["l3"] ** 2 / 12,
    "P": MASS["P"] * (LENGTH["l3"] + LENGTH["l4"]) ** 2 / 12,
    "Q": MASS["Q"] * LENGTH["l5"] ** 2 / 12,
}

DEFAULT_PASSIVE_GUESS = np.deg2rad((80.0, 60.0, 0.0, 30.0, 30.0, 90.0))


def _tree_adjacency() -> dict[str, list[str]]:
    adj = {vertex: [] for vertex in VERTICES}
    for a, b in TREE_EDGES:
        adj[a].append(b)
        adj[b].append(a)
    return adj


TREE_ADJ = _tree_adjacency()


def _tree_path(start: str, finish: str) -> list[str]:
    queue = deque((start,))
    previous: dict[str, str | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == finish:
            break
        for neighbor in TREE_ADJ[current]:
            if neighbor not in previous:
                previous[neighbor] = current
                queue.append(neighbor)
    if finish not in previous:
        raise ValueError(f"No hay camino en el árbol entre {start} y {finish}.")
    path = []
    current: str | None = finish
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _cycle_from_chord(start: str, chord_from: str, chord_to: str) -> tuple[str, ...]:
    """Crea un ciclo al insertar una cuerda orientada en el árbol recubridor."""
    left = _tree_path(start, chord_from)
    right = _tree_path(chord_to, start)
    return tuple(left + right)


# La dirección solo fija el signo del cierre. Cada ciclo se construye
# algorítmicamente con un camino del árbol y una cuerda.
CYCLES = (
    _cycle_from_chord("T", "P", "L"),
    _cycle_from_chord("T", "N", "M"),
    _cycle_from_chord("J", "P", "Q"),
)
CYCLE_NAMES = ("T-J-P-L-K-T", "T-J-N-M-T", "J-P-Q-O-N-J")


@dataclass(frozen=True)
class CycleSegment:
    body: str
    local_vector: np.ndarray


def _segments_for_cycle(cycle: tuple[str, ...]) -> tuple[CycleSegment, ...]:
    segments = []
    for i, body in enumerate(cycle[:-1]):
        previous = cycle[i - 1] if i else cycle[-2]
        following = cycle[i + 1]
        entering = edge_key(previous, body)
        leaving = edge_key(body, following)
        vector = PIN[body][leaving] - PIN[body][entering]
        segments.append(CycleSegment(body, vector))
    return tuple(segments)


CYCLE_SEGMENTS = tuple(_segments_for_cycle(cycle) for cycle in CYCLES)


def adjacency_matrix() -> list[list[int]]:
    matrix = [[0 for _ in VERTICES] for _ in VERTICES]
    index = {v: i for i, v in enumerate(VERTICES)}
    for a, b in EDGES:
        matrix[index[a]][index[b]] = 1
        matrix[index[b]][index[a]] = 1
    return matrix


def closure_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve seis cierres escalares y su jacobiano respecto a ocho ángulos."""
    q = np.asarray(q, dtype=float)
    phi = np.zeros((len(CYCLE_SEGMENTS), 2), dtype=float)
    jac = np.zeros((len(CYCLE_SEGMENTS), 2, len(BODY_ORDER)), dtype=float)
    for k, segments in enumerate(CYCLE_SEGMENTS):
        for segment in segments:
            body = segment.body
            if body == "T":
                angle = 0.0
            else:
                angle = q[BODY_INDEX[body]]
            world_vector = rotation(angle) @ segment.local_vector
            phi[k] += world_vector
            if body != "T":
                jac[k, :, BODY_INDEX[body]] += quarter_turn(world_vector)
    return phi.reshape(-1), jac.reshape(2 * len(CYCLE_SEGMENTS), len(BODY_ORDER))


def closure_gamma(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Término centrípeto de la segunda derivada de los cierres."""
    gamma = np.zeros((len(CYCLE_SEGMENTS), 2), dtype=float)
    for k, segments in enumerate(CYCLE_SEGMENTS):
        for segment in segments:
            body = segment.body
            if body == "T":
                continue
            i = BODY_INDEX[body]
            world_vector = rotation(q[i]) @ segment.local_vector
            gamma[k] -= qd[i] ** 2 * world_vector
    return gamma.reshape(-1)


def legacy_closures(q: np.ndarray) -> np.ndarray:
    """Cierres escritos explícitamente en el modelo previo, para comparación."""
    q = np.asarray(q, dtype=float)
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8 = q
    r_ab = _unit(theta1, LENGTH["l2"])
    r_ac = _unit(theta2, LENGTH["l1"])
    r_cd = _unit(theta3, LENGTH["l3"])
    r_bd = _unit(theta7, LENGTH["l4"])
    r_ib = _unit(theta7, LENGTH["l3"])
    r_ih = _unit(theta8, LENGTH["l5"])
    r_hg = _unit(theta6, LENGTH["l3"])
    r_bg = _unit(theta5 + 95.0 * pi / 180.0, LENGTH["l6"])
    r_bf = _unit(theta5, LENGTH["l7"])
    r_ef = _unit(theta4, LENGTH["l3"])
    r_ae = _vec(LENGTH["l10"], LENGTH["l9"])
    loops = (
        r_ab + r_bd - r_cd - r_ac,
        r_ab + r_bf - r_ef - r_ae,
        -r_ib + r_ih + r_hg - r_bg,
    )
    return np.asarray(loops).reshape(-1)


def solve_configuration(
    active_angles: Iterable[float],
    guess: np.ndarray | None = None,
) -> np.ndarray:
    """Adaptación de FK del taller: fija dos ángulos y minimiza el cierre."""
    active = np.asarray(tuple(active_angles), dtype=float)
    if active.shape != (2,):
        raise ValueError("Se esperan dos ángulos de entrada.")
    initial = DEFAULT_PASSIVE_GUESS.copy() if guess is None else np.asarray(guess, dtype=float).copy()

    def objective(passive: np.ndarray) -> tuple[float, np.ndarray]:
        q = np.r_[active, passive]
        loop_error, jac = closure_jacobian(q)
        cost = float(loop_error @ loop_error)
        gradient = 2.0 * jac[:, 2:].T @ loop_error
        return cost, gradient

    result = minimize(
        objective,
        x0=initial,
        jac=True,
        method="BFGS",
        options={"maxiter": 10_000, "gtol": 1e-12, "disp": False},
    )
    error = float(np.sqrt(max(0.0, result.fun)))
    if not result.success and error > 1e-8:
        raise ValueError(
            "La cinemática directa no convergió para esos dos ángulos. "
            f"Residuo de cierre {error:.3g} m."
        )
    return np.r_[active, result.x]


def compatible_velocity(q: np.ndarray, active_velocity: Iterable[float]) -> np.ndarray:
    active = np.asarray(tuple(active_velocity), dtype=float)
    if active.shape != (2,):
        raise ValueError("Se esperan dos velocidades de entrada.")
    _, jac = closure_jacobian(q)
    qd = np.zeros(len(BODY_ORDER), dtype=float)
    qd[:2] = active
    qd[2:] = np.linalg.solve(jac[:, 2:], -(jac[:, :2] @ active))
    return qd


def compatible_acceleration(
    q: np.ndarray,
    qd: np.ndarray,
    active_acceleration: Iterable[float],
) -> np.ndarray:
    active = np.asarray(tuple(active_acceleration), dtype=float)
    if active.shape != (2,):
        raise ValueError("Se esperan dos aceleraciones de entrada.")
    _, jac = closure_jacobian(q)
    qdd = np.zeros(len(BODY_ORDER), dtype=float)
    qdd[:2] = active
    rhs = -(jac[:, :2] @ active + closure_gamma(q, qd))
    qdd[2:] = np.linalg.solve(jac[:, 2:], rhs)
    return qdd


def body_kinematics(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Calcula poses, velocidades y aceleraciones desde las aristas del árbol."""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    data: dict[str, dict[str, np.ndarray | float]] = {
        "T": {
            "angle": 0.0, "omega": 0.0, "alpha": 0.0,
            "origin": _vec(0, 0), "velocity": _vec(0, 0), "acceleration": _vec(0, 0),
        }
    }
    for body in BODY_ORDER:
        data[body] = {
            "angle": float(q[BODY_INDEX[body]]),
            "omega": float(qd[BODY_INDEX[body]]),
            "alpha": float(qdd[BODY_INDEX[body]]),
        }

    for parent, child in TREE_PARENT_CHILD:
        edge = edge_key(parent, child)
        parent_data = data[parent]
        child_data = data[child]
        parent_angle = float(parent_data["angle"])
        child_angle = float(child_data["angle"])
        parent_omega = float(parent_data["omega"])
        child_omega = float(child_data["omega"])
        parent_alpha = float(parent_data["alpha"])
        child_alpha = float(child_data["alpha"])
        parent_r = rotation(parent_angle) @ PIN[parent][edge]
        child_r = rotation(child_angle) @ PIN[child][edge]
        parent_origin = np.asarray(parent_data["origin"])
        parent_velocity = np.asarray(parent_data["velocity"])
        parent_acceleration = np.asarray(parent_data["acceleration"])
        joint_position = parent_origin + parent_r
        joint_velocity = parent_velocity + parent_omega * quarter_turn(parent_r)
        joint_acceleration = (
            parent_acceleration
            + parent_alpha * quarter_turn(parent_r)
            - parent_omega**2 * parent_r
        )
        child_data["origin"] = joint_position - child_r
        child_data["velocity"] = joint_velocity - child_omega * quarter_turn(child_r)
        child_data["acceleration"] = (
            joint_acceleration
            - child_alpha * quarter_turn(child_r)
            + child_omega**2 * child_r
        )

    for body in BODY_ORDER:
        item = data[body]
        center_local = COM_LOCAL[body]
        center_r = rotation(float(item["angle"])) @ center_local
        item["center"] = np.asarray(item["origin"]) + center_r
        item["center_velocity"] = np.asarray(item["velocity"]) + float(item["omega"]) * quarter_turn(center_r)
        item["center_acceleration"] = (
            np.asarray(item["acceleration"])
            + float(item["alpha"]) * quarter_turn(center_r)
            - float(item["omega"]) ** 2 * center_r
        )
    return data


def all_joint_positions(
    q: np.ndarray,
    poses: dict[str, dict[str, np.ndarray | float]] | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    if poses is None:
        poses = body_kinematics(q, np.zeros(8), np.zeros(8))
    result = {}
    max_gap = 0.0
    for a, b in EDGES:
        edge = edge_key(a, b)
        pa = np.asarray(poses[a]["origin"]) + rotation(float(poses[a]["angle"])) @ PIN[a][edge]
        pb = np.asarray(poses[b]["origin"]) + rotation(float(poses[b]["angle"])) @ PIN[b][edge]
        result[a + b] = 0.5 * (pa + pb)
        max_gap = max(max_gap, float(np.linalg.norm(pa - pb)))
    return result, max_gap


def _dynamic_rhs(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray | float]]]:
    poses = body_kinematics(q, qd, qdd)
    rhs = []
    for body in BODY_ORDER:
        item = poses[body]
        acceleration = np.asarray(item["center_acceleration"])
        rhs.extend((
            MASS[body] * acceleration[0],
            MASS[body] * acceleration[1] + MASS[body] * GRAVITY,
            INERTIA[body] * float(item["alpha"]),
        ))
    return np.asarray(rhs, dtype=float), poses


def _force_matrix(
    q: np.ndarray,
    poses: dict[str, dict[str, np.ndarray | float]],
) -> np.ndarray:
    """Newton-Euler coefficients for the two components at every graph edge."""
    matrix = np.zeros((3 * len(BODY_ORDER), 2 * len(EDGES)), dtype=float)
    joint_positions, _ = all_joint_positions(q, poses)
    for e_index, (a, b) in enumerate(EDGES):
        point = joint_positions[a + b]
        for body, sign in ((a, -1.0), (b, 1.0)):
            if body == "T":
                continue
            row = 3 * BODY_INDEX[body]
            center = np.asarray(poses[body]["center"])
            arm = point - center
            col_x, col_y = 2 * e_index, 2 * e_index + 1
            matrix[row, col_x] += sign
            matrix[row + 1, col_y] += sign
            matrix[row + 2, col_x] += -sign * arm[1]
            matrix[row + 2, col_y] += sign * arm[0]
    return matrix


def _actuator_matrix() -> np.ndarray:
    matrix = np.zeros((3 * len(BODY_ORDER), 2), dtype=float)
    matrix[3 * BODY_INDEX["J"] + 2, 0] = 1.0
    matrix[3 * BODY_INDEX["K"] + 2, 1] = 1.0
    return matrix


ACTUATOR_MATRIX = _actuator_matrix()


def inverse_dynamics(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> dict[str, object]:
    """Resuelve fuerzas de articulación y los dos pares de entrada."""
    rhs, poses = _dynamic_rhs(q, qd, qdd)
    force = _force_matrix(q, poses)
    solution = np.linalg.solve(np.column_stack((force, ACTUATOR_MATRIX)), rhs)
    forces = solution[: 2 * len(EDGES)].reshape(len(EDGES), 2)
    torques = solution[2 * len(EDGES) :]
    residual = np.column_stack((force, ACTUATOR_MATRIX)) @ solution - rhs
    return {
        "torques": torques,
        "forces": {a + b: forces[i].tolist() for i, (a, b) in enumerate(EDGES)},
        "newton_euler_residual": float(np.max(np.abs(residual))),
    }


def forward_dynamics(
    q: np.ndarray,
    qd: np.ndarray,
    torques: Iterable[float],
    stabilization_frequency: float = 0.0,
    damping_ratio: float = 0.8,
) -> dict[str, object]:
    """Resuelve aceleraciones y reacciones dadas dos entradas de torque.

    La matriz del sistema reúne 24 balances de Newton-Euler y seis derivadas
    segundas de los cierres de ciclo.
    """
    torques = np.asarray(tuple(torques), dtype=float)
    if torques.shape != (2,):
        raise ValueError("Se esperan dos torques de entrada en N m.")
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)

    zero_qdd = np.zeros(8)
    rhs0, poses = _dynamic_rhs(q, qd, zero_qdd)
    force = _force_matrix(q, poses)
    mass_acceleration_columns = np.empty((3 * len(BODY_ORDER), len(BODY_ORDER)))
    for i in range(len(BODY_ORDER)):
        unit_qdd = np.zeros(8)
        unit_qdd[i] = 1.0
        rhs_i, _ = _dynamic_rhs(q, qd, unit_qdd)
        mass_acceleration_columns[:, i] = rhs_i - rhs0

    phi, jac = closure_jacobian(q)
    gamma = closure_gamma(q, qd)
    constraint_rhs = -gamma
    if stabilization_frequency > 0:
        velocity_error = jac @ qd
        constraint_rhs -= (
            2 * damping_ratio * stabilization_frequency * velocity_error
            + stabilization_frequency**2 * phi
        )

    system = np.zeros((24 + 6, 8 + 22), dtype=float)
    system[:24, :8] = -mass_acceleration_columns
    system[:24, 8:] = force
    system[24:, :8] = jac
    rhs = np.r_[rhs0 - ACTUATOR_MATRIX @ torques, constraint_rhs]
    solution = np.linalg.solve(system, rhs)
    qdd = solution[:8]
    forces = solution[8:].reshape(len(EDGES), 2)
    residual = system @ solution - rhs
    return {
        "qdd": qdd,
        "forces": {a + b: forces[i].tolist() for i, (a, b) in enumerate(EDGES)},
        "linear_system_residual": float(np.max(np.abs(residual))),
        "position_residual": float(np.linalg.norm(phi)),
        "velocity_residual": float(np.linalg.norm(jac @ qd)),
        "acceleration_residual": float(np.linalg.norm(jac @ qdd + gamma)),
    }


def forward_dynamics_two_dof(
    q: np.ndarray,
    qd: np.ndarray,
    torques: Iterable[float],
) -> dict[str, object]:
    """Resuelve dinámica directa con solo theta1 y theta2 independientes."""
    torques = np.asarray(tuple(torques), dtype=float)
    if torques.shape != (2,):
        raise ValueError("Se esperan dos torques de entrada en N m.")
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    rhs0, poses = _dynamic_rhs(q, qd, np.zeros(8))
    force = _force_matrix(q, poses)

    inertia_map = np.empty((24, 8))
    for i in range(8):
        unit_qdd = np.zeros(8)
        unit_qdd[i] = 1.0
        rhs_i, _ = _dynamic_rhs(q, qd, unit_qdd)
        inertia_map[:, i] = rhs_i - rhs0

    phi, jac = closure_jacobian(q)
    gamma = closure_gamma(q, qd)
    jac_passive = jac[:, 2:]
    passive_from_active = np.linalg.solve(jac_passive, -jac[:, :2])
    passive_bias = np.linalg.solve(jac_passive, -gamma)

    reduced_inertia = inertia_map[:, :2] + inertia_map[:, 2:] @ passive_from_active
    bias_rhs = rhs0 + inertia_map[:, 2:] @ passive_bias - ACTUATOR_MATRIX @ torques
    system = np.column_stack((-reduced_inertia, force))
    solution = np.linalg.solve(system, bias_rhs)

    qdd = np.r_[solution[:2], passive_bias + passive_from_active @ solution[:2]]
    forces = solution[2:].reshape(len(EDGES), 2)
    equation_residual = system @ solution - bias_rhs
    acceleration_residual = float(np.linalg.norm(jac @ qdd + gamma))
    return {
        "qdd": qdd,
        "forces": {a + b: forces[i].tolist() for i, (a, b) in enumerate(EDGES)},
        "linear_system_residual": float(np.max(np.abs(equation_residual))),
        "position_residual": float(np.linalg.norm(phi)),
        "velocity_residual": float(np.linalg.norm(jac @ qd)),
        "acceleration_residual": acceleration_residual,
    }


def body_paths() -> dict[str, list[str]]:
    """Aristas articulares que delimitan cada eslabón para dibujarlo."""
    return {
        "J": ["TJ", "JP"],
        "K": ["TK", "KL"],
        "L": ["KL", "LP"],
        "M": ["TM", "MN"],
        "N": ["JN", "NO", "MN"],
        "O": ["NO", "OQ"],
        "P": ["JP", "LP", "QP"],
        "Q": ["QP", "OQ"],
    }


def graph_payload() -> dict[str, object]:
    return {
        "vertices": list(VERTICES),
        "edges": [list(edge) for edge in EDGES],
        "tree_edges": [list(edge) for edge in TREE_EDGES],
        "chords": [list(edge) for edge in CHORDS],
        "cycles": [list(cycle) for cycle in CYCLES],
        "cycle_names": list(CYCLE_NAMES),
        "adjacency": adjacency_matrix(),
        "body_paths": body_paths(),
    }


def pose_payload(q: np.ndarray) -> dict[str, object]:
    poses = body_kinematics(q, np.zeros(8), np.zeros(8))
    joints, joint_gap = all_joint_positions(q, poses)
    return {
        "joint_positions": {name: position.tolist() for name, position in joints.items()},
        "joint_gap": joint_gap,
        "closure_norm": float(np.linalg.norm(closure_jacobian(q)[0])),
    }


def simulate_kinematic_routine(
    routine: str,
    samples: int = 120,
) -> dict[str, object]:
    """Reproduce las dos rutinas de ángulos del taller original."""
    samples = int(np.clip(samples, 20, 240))
    t_values = np.linspace(0.0, 2.0 * pi, samples)

    if routine == "animation1":
        theta1_min, theta1_max = np.deg2rad((90.0, 160.0))
        theta2_min, theta2_max = np.deg2rad((0.0, 90.0))
        theta1_values = theta1_min + (theta1_max - theta1_min) * 0.5 * (1.0 - np.cos(t_values / 2.0))
        theta2_values = theta2_min + (theta2_max - theta2_min) * 0.5 * (1.0 - np.cos(t_values))
        title = "Rutina 1 · θ₁ y θ₂ del taller"
    elif routine == "animation2":
        theta1_values = pi / 2.0 + (pi / 18.0) * np.sin((2.0 * pi / 3.0) * t_values)
        theta2_values = np.zeros_like(t_values)
        title = "Rutina 2 · θ₂ fijo del taller"
    else:
        raise ValueError("Rutina desconocida. Elige animation1 o animation2.")

    frames = []
    passive_guess = DEFAULT_PASSIVE_GUESS.copy()
    max_closure = 0.0
    max_joint_gap = 0.0
    for i, time_value in enumerate(t_values):
        q = solve_configuration((theta1_values[i], theta2_values[i]), guess=passive_guess)
        passive_guess = q[2:].copy()
        pose = pose_payload(q)
        max_closure = max(max_closure, float(pose["closure_norm"]))
        max_joint_gap = max(max_joint_gap, float(pose["joint_gap"]))
        frames.append({
            "time": float(time_value),
            "time_label": f"t = {time_value:.2f}",
            "pose_label": f"{title} · {i + 1}/{samples}",
            "q_deg": np.rad2deg(q).tolist(),
            "joint_positions": pose["joint_positions"],
        })

    return {
        "routine": routine,
        "title": title,
        "frames": frames,
        "max_closure_norm": max_closure,
        "max_joint_gap": max_joint_gap,
    }


def simulate_forward(
    active_angles_deg: Iterable[float],
    active_velocities_deg_s: Iterable[float],
    torques_nm: Iterable[float],
    duration_s: float = 0.6,
    samples: int = 61,
) -> dict[str, object]:
    """Integra los dos grados de libertad y reconstruye el resto por cierres."""
    duration_s = float(duration_s)
    if not 0.05 <= duration_s <= 1.5:
        raise ValueError("La duración debe estar entre 0.05 y 1.5 segundos.")
    samples = int(np.clip(samples, 20, 100))
    active0 = np.deg2rad(np.asarray(tuple(active_angles_deg), dtype=float))
    velocity0 = np.deg2rad(np.asarray(tuple(active_velocities_deg_s), dtype=float))
    torque = np.asarray(tuple(torques_nm), dtype=float)
    if active0.shape != (2,) or velocity0.shape != (2,) or torque.shape != (2,):
        raise ValueError("Se esperan dos ángulos, dos velocidades y dos torques.")

    q0 = solve_configuration(active0)
    y0 = np.r_[active0, velocity0]
    continuation = {"passive": q0[2:].copy()}

    def state_derivative(_time: float, y: np.ndarray) -> np.ndarray:
        q = solve_configuration(y[:2], guess=continuation["passive"])
        continuation["passive"] = q[2:].copy()
        qd = compatible_velocity(q, y[2:])
        direct = forward_dynamics_two_dof(q, qd, torque)
        return np.r_[y[2:], np.asarray(direct["qdd"])[:2]]

    times = np.linspace(0.0, duration_s, samples)
    result = solve_ivp(
        state_derivative,
        (0.0, duration_s),
        y0,
        method="RK45",
        t_eval=times,
        max_step=duration_s / 15.0,
        rtol=2e-6,
        atol=1e-8,
    )
    if not result.success:
        raise ValueError(f"La integración se detuvo: {result.message}")

    frames = []
    frame_guess = q0[2:].copy()
    max_closure = 0.0
    max_joint_gap = 0.0
    max_velocity_constraint = 0.0
    for i, time_s in enumerate(result.t):
        active = result.y[:2, i]
        active_velocity = result.y[2:, i]
        q = solve_configuration(active, guess=frame_guess)
        frame_guess = q[2:].copy()
        qd = compatible_velocity(q, active_velocity)
        data = pose_payload(q)
        _, jac = closure_jacobian(q)
        max_closure = max(max_closure, data["closure_norm"])
        max_joint_gap = max(max_joint_gap, data["joint_gap"])
        max_velocity_constraint = max(max_velocity_constraint, float(np.linalg.norm(jac @ qd)))
        frames.append({
            "time": float(time_s),
            "q_deg": np.rad2deg(q).tolist(),
            "joint_positions": data["joint_positions"],
        })
    return {
        "frames": frames,
        "duration": duration_s,
        "integrator_evaluations": int(result.nfev),
        "max_closure_norm": max_closure,
        "max_joint_gap": max_joint_gap,
        "max_velocity_constraint": max_velocity_constraint,
        "torques_nm": torque.tolist(),
    }


def default_configuration() -> dict[str, object]:
    """Pose inicial usada por la página y los chequeos del modelo."""
    q = solve_configuration(np.deg2rad((90.0, 0.0)))
    return {
        "q_rad": q.tolist(),
        "q_deg": np.rad2deg(q).tolist(),
        **pose_payload(q),
    }
