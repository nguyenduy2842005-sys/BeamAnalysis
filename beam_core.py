from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ==========================================================
# DATA MODELS
# ==========================================================

@dataclass
class BeamInput:
    length: float
    beam_type: str = "simple"          # "simple" | "cantilever"
    uvl_type: str = "increase"         # "increase" | "decrease"
    point_loads: list[tuple[float, float]] = field(default_factory=list)       # (P, x)
    point_moments: list[tuple[float, float]] = field(default_factory=list)     # (M, x)
    udls: list[tuple[float, float, float]] = field(default_factory=list)       # (q, x1, x2)
    uvls: list[tuple[float, float, float]] = field(default_factory=list)       # (qmax, x1, x2)


@dataclass
class BeamResult:
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    theta: np.ndarray
    deflection: np.ndarray

    r1: float
    r2: float

    rv_fixed: float
    mr_fixed: float

    report: str


# ==========================================================
# SAFE HELPERS
# ==========================================================

def _as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_point_steps(raw) -> list[dict]:
    """
    Chuẩn hóa point_steps về list dict:
    [{"P":..., "x":..., "M":...}, ...]
    """
    out = []
    if raw is None:
        return out

    for item in raw:
        if isinstance(item, dict):
            P = _as_float(item.get("P", 0.0))
            x = _as_float(item.get("x", 0.0))
            M = _as_float(item.get("M", P * x))
        else:
            # tuple/list/ndarray hoặc object khác
            try:
                seq = list(item)
            except TypeError:
                # item là float -> bỏ qua thay vì crash
                continue

            if len(seq) >= 3:
                P = _as_float(seq[0])
                x = _as_float(seq[1])
                M = _as_float(seq[2], P * x)
            elif len(seq) == 2:
                P = _as_float(seq[0])
                x = _as_float(seq[1])
                M = P * x
            else:
                continue

        out.append({"P": P, "x": x, "M": M})
    return out


def _normalize_udl_steps(raw) -> list[dict]:
    """
    Chuẩn hóa udl_steps về list dict:
    [{"q":..., "L":..., "F":..., "xc":..., "M":...}, ...]
    """
    out = []
    if raw is None:
        return out

    for item in raw:
        if isinstance(item, dict):
            q = _as_float(item.get("q", 0.0))
            L = _as_float(item.get("L", 0.0))
            F = _as_float(item.get("F", q * L))
            xc = _as_float(item.get("xc", 0.0))
            M = _as_float(item.get("M", F * xc))
        else:
            try:
                seq = list(item)
            except TypeError:
                continue

            # hỗ trợ (q, a, b) hoặc (q, L, F, xc, M)
            if len(seq) >= 5:
                q = _as_float(seq[0])
                L = _as_float(seq[1])
                F = _as_float(seq[2])
                xc = _as_float(seq[3])
                M = _as_float(seq[4])
            elif len(seq) == 3:
                q = _as_float(seq[0])
                a = _as_float(seq[1])
                b = _as_float(seq[2])
                L = b - a
                F = q * L
                xc = a + L / 2
                M = F * xc
            else:
                continue

        out.append({"q": q, "L": L, "F": F, "xc": xc, "M": M})
    return out


def _normalize_uvl_steps(raw, uvl_type: str = "increase") -> list[dict]:
    """
    Chuẩn hóa uvl_steps về list dict:
    [{"qmax":..., "a":..., "b":..., "L":..., "F":..., "xc":..., "M":..., "type":...}, ...]
    """
    out = []
    if raw is None:
        return out

    for item in raw:
        if isinstance(item, dict):
            qmax = _as_float(item.get("qmax", 0.0))
            a = _as_float(item.get("a", 0.0))
            b = _as_float(item.get("b", 0.0))
            L = _as_float(item.get("L", b - a))
            F = _as_float(item.get("F", 0.5 * qmax * L))
            xc = _as_float(item.get("xc", 0.0))
            M = _as_float(item.get("M", F * xc))
            typ = item.get("type", uvl_type)
        else:
            try:
                seq = list(item)
            except TypeError:
                continue

            # hỗ trợ (qmax, a, b)
            if len(seq) >= 3:
                qmax = _as_float(seq[0])
                a = _as_float(seq[1])
                b = _as_float(seq[2])
                L = b - a
                F = 0.5 * qmax * L
                if uvl_type == "increase":
                    xc = a + 2 * L / 3
                else:
                    xc = a + L / 3
                M = F * xc
                typ = uvl_type
            else:
                continue

        out.append({
            "qmax": qmax,
            "a": a,
            "b": b,
            "L": L,
            "F": F,
            "xc": xc,
            "M": M,
            "type": typ,
        })
    return out


def _normalize_moment_steps(raw) -> list[dict]:
    """
    Chuẩn hóa moment_steps về list dict:
    [{"M":..., "x":...}, ...]
    """
    out = []
    if raw is None:
        return out

    for item in raw:
        if isinstance(item, dict):
            M = _as_float(item.get("M", 0.0))
            x = _as_float(item.get("x", 0.0))
        else:
            try:
                seq = list(item)
            except TypeError:
                continue

            if len(seq) >= 2:
                M = _as_float(seq[0])
                x = _as_float(seq[1])
            else:
                continue

        out.append({"M": M, "x": x})
    return out


# ==========================================================
# SOLVER
# ==========================================================

def solve_beam(data: BeamInput, step: float = 0.01) -> BeamResult:
    """
    Single Beam solver chuẩn theo logic bản cũ (MATLAB port)
    - Dầm tựa đơn: phản lực hướng lên mang dấu âm
    - Dầm console: ngàm bên phải tại x = L
    - Tải trọng hướng xuống mang dấu dương, tích lũy bằng toán tử `+=`
    """
    rv_fixed = 0.0
    mr_fixed = 0.0
    r1 = 0.0
    r2 = 0.0

    l = float(data.length)
    if l <= 0:
        raise ValueError("Chiều dài dầm phải lớn hơn 0.")

    # Chuyển đổi dữ liệu sang mảng numpy
    p = np.array(data.point_loads, dtype=float).reshape(-1, 2) if data.point_loads else np.empty((0, 2))
    m = np.array(data.point_moments, dtype=float).reshape(-1, 2) if data.point_moments else np.empty((0, 2))
    udl = np.array(data.udls, dtype=float).reshape(-1, 3) if data.udls else np.empty((0, 3))
    uvl = np.array(data.uvls, dtype=float).reshape(-1, 3) if data.uvls else np.empty((0, 3))

    cp, cm, cudl, cuvl = len(p), len(m), len(udl), len(uvl)

    x = np.arange(0.0, l + step / 2, step)
    shear = np.zeros_like(x)
    moment = np.zeros_like(x)
    theta_int = np.zeros_like(x)
    w_int = np.zeros_like(x)

    sum_p_mom = sum_udl_mom = sum_uvl_mom = sum_m_mom = 0.0
    sum_p_val = sum_udl_val = sum_uvl_val = 0.0
    is_simple = data.beam_type == "simple"

    # ------------------------------------------------------
    # TÍNH TOÁN PHẢN LỰC GỐI / NGÀM
    # ------------------------------------------------------
    if is_simple:
        if cp > 0:
            sum_p_mom = float(np.sum(p[:, 0] * p[:, 1]))
            sum_p_val = float(np.sum(p[:, 0]))
        if cudl > 0:
            len_udl = udl[:, 2] - udl[:, 1]
            val_udl = udl[:, 0] * len_udl
            pos_udl = udl[:, 1] + len_udl / 2
            sum_udl_mom = float(np.sum(val_udl * pos_udl))
            sum_udl_val = float(np.sum(val_udl))
        if cuvl > 0:
            for q_max, a, b in uvl:
                span = b - a
                val = 0.5 * q_max * span
                pos = a + 2 * span / 3 if data.uvl_type == "increase" else a + span / 3
                sum_uvl_mom += val * pos
                sum_uvl_val += val
        if cm > 0:
            sum_m_mom = float(np.sum(m[:, 0]))

        r2 = -(sum_p_mom + sum_udl_mom + sum_uvl_mom + sum_m_mom) / l
        r1 = -(sum_p_val + sum_udl_val + sum_uvl_val + r2)

        # Khởi tạo biểu đồ với phản lực nút đầu gối A
        shear += r1
        moment += r1 * x
    else:
        r1 = 0.0
        r2 = 0.0
        total_vertical_load = (
                sum(load for load, _ in data.point_loads)
                + sum(q * (b - a) for q, a, b in data.udls)
                + sum(0.5 * q * (b - a) for q, a, b in data.uvls)
        )
        rv_fixed = -total_vertical_load

        moment_p = sum(load * (l - pos) for load, pos in data.point_loads)
        moment_udl = sum(q * (b - a) * (l - (a + (b - a) / 2)) for q, a, b in data.udls)
        moment_uvl = 0.0
        for q, a, b in data.uvls:
            span = b - a
            xr = a + 2 * span / 3 if data.uvl_type == "increase" else a + span / 3
            moment_uvl += 0.5 * q * span * (l - xr)
        moment_m = sum(mi for mi, _ in data.point_moments)
        mr_fixed = -(moment_p + moment_udl + moment_uvl + moment_m)

    # Tích phân ban đầu từ phản lực R1
    theta_int += (r1 * x ** 2) / 2
    w_int += (r1 * x ** 3) / 6

    # ------------------------------------------------------
    # TÍNH TOÁN NỘI LỰC THEO LOGIC TÍCH LŨY (+=) BAN ĐẦU
    # ------------------------------------------------------

    # 1) POINT LOADS
    for load, a in p:
        mask = x >= a
        shear[mask] += load
        moment[mask] += load * (x[mask] - a)
        theta_int[mask] += load * (x[mask] - a) ** 2 / 2
        w_int[mask] += load * (x[mask] - a) ** 3 / 6

    # 2) UDL
    for q, a, b in udl:
        mask_in = (x > a) & (x <= b)
        mask_after = x > b
        shear[mask_in] += q * (x[mask_in] - a)
        moment[mask_in] += 0.5 * q * (x[mask_in] - a) ** 2
        shear[mask_after] += q * (b - a)
        moment[mask_after] += q * (b - a) * (x[mask_after] - (a + b) / 2)

        m_a = x > a;
        m_b = x > b
        theta_int[m_a] += (q / 6) * (x[m_a] - a) ** 3
        w_int[m_a] += (q / 24) * (x[m_a] - a) ** 4
        theta_int[m_b] -= (q / 6) * (x[m_b] - b) ** 3
        w_int[m_b] -= (q / 24) * (x[m_b] - b) ** 4

    # 3) UVL
    for q_max, a, b in uvl:
        span = b - a
        if span <= 0:
            continue
        xx = np.maximum(x - a, 0)
        xb = np.maximum(x - b, 0)
        if data.uvl_type == "increase":
            shear += (q_max / (2 * span)) * (xx ** 2 - xb ** 2) - (0.5 * q_max * span) * (x > b)
            moment += (q_max / (6 * span)) * (xx ** 3 - xb ** 3) - (0.5 * q_max * span) * xb - (
                        q_max * span ** 2 / 6) * (x > b)

            m_a = x > a;
            m_b = x > b
            theta_int[m_a] += (q_max / (24 * span)) * (x[m_a] - a) ** 4
            w_int[m_a] += (q_max / (120 * span)) * (x[m_a] - a) ** 5
            theta_int[m_b] -= (q_max / (24 * span)) * (x[m_b] - b) ** 4 + (q_max * span / 6) * (x[m_b] - b) ** 3
            w_int[m_b] -= (q_max / (120 * span)) * (x[m_b] - b) ** 5 + (q_max * span / 24) * (x[m_b] - b) ** 4
        else:
            shear += q_max * xx - (q_max / (2 * span)) * xx ** 2 - (0.5 * q_max * span) * (x > b)
            moment += (q_max / 2) * xx ** 2 - (q_max / (6 * span)) * xx ** 3 - (
                        0.5 * q_max * span * xb + q_max * span ** 2 / 3) * (x > b)
            # Lưu ý: Phần tính toán võng/xoay của UVL decrease bản cũ chưa đầy đủ, giữ nguyên như cũ.

    # 4) POINT MOMENTS
    for mi, a in m:
        mask = x >= a
        moment[mask] -= mi
        theta_int[mask] -= mi * (x[mask] - a)
        w_int[mask] -= mi * (x[mask] - a) ** 2 / 2

    # ------------------------------------------------------
    # HẰNG SỐ TÍCH PHÂN & ĐỘ VÕNG
    # ------------------------------------------------------
    if is_simple:
        c2 = 0.0
        c1 = -w_int[-1] / l
    else:
        c1 = -theta_int[-1]
        c2 = -(w_int[-1] + c1 * l)

    deflection = -(w_int + c1 * x + c2)
    theta = theta_int + c1

    # Làm sạch các giá trị quá nhỏ về 0
    shear[np.abs(shear) < 1e-10] = 0.0
    moment[np.abs(moment) < 1e-10] = 0.0
    theta[np.abs(theta) < 1e-12] = 0.0
    deflection[np.abs(deflection) < 1e-12] = 0.0

    # ------------------------------------------------------
    # XÂY DỰNG REPORT (Bên trong hàm solve_beam)
    # ------------------------------------------------------
    report_lines = []
    report_lines += build_header_report(data)
    report_lines += build_load_report(data)

    if is_simple:
        report_lines += build_reaction_report_fixed(data, r1, r2, sum_p_mom, sum_udl_mom, sum_uvl_mom, sum_m_mom)
    else:
        report_lines += build_fixed_report_fixed(data, rv_fixed, mr_fixed)

    report_lines += build_shear_report(data, x, shear)
    report_lines += build_moment_report(data, x, moment, shear)

    # SỬA DÒNG BÊN DƯỚI: Truyền thêm c1, c2, w_int[-1], theta_int[-1] để làm thuyết minh "Thế số"
    report_lines += build_deflection_report(data, x, deflection, c1, c2, float(w_int[-1]), float(theta_int[-1]))

    report = "\n".join(report_lines)

    return BeamResult(
        x=x, shear=shear, moment=moment, theta=theta, deflection=deflection,
        r1=float(r1), r2=float(r2), rv_fixed=float(rv_fixed), mr_fixed=float(mr_fixed),
        report=report
    )

def build_header_report(data: BeamInput) -> list[str]:
    lines = []
    lines.append("=========== THUYẾT MINH TÍNH TOÁN ===========")
    lines.append(f"Chiều dài dầm: {data.length:.2f} m")
    lines.append("Loại dầm: Dầm tựa đơn" if data.beam_type == "simple" else "Loại dầm: Dầm console")
    lines.append("Phương trình cân bằng:")
    lines.append("ΣFx = 0")
    lines.append("ΣFy = 0")
    lines.append("ΣM = 0")
    return lines


def build_load_report(data: BeamInput) -> list[str]:
    lines = []
    lines.append("==============================")
    lines.append("KHAI BÁO TẢI TRỌNG")
    lines.append("==============================")

    # --- Tải tập trung ---
    lines.append("TẢI TẬP TRUNG")
    if data.point_loads:
        for i, (P, x) in enumerate(data.point_loads, 1):
            lines.append(f"P{i} = {P:.2f} kN tại x = {x:.2f} m")
    else:
        lines.append("Không có")

    # --- Tải phân bố đều ---
    lines.append("TẢI PHÂN BỐ ĐỀU")
    if data.udls:
        for i, (q, a, b) in enumerate(data.udls, 1):
            lines.append(f"UDL{i}: q = {q:.2f} kN/m, từ x = {a:.2f} đến x = {b:.2f} m")
    else:
        lines.append("Không có")

    # --- Tải tam giác ---
    lines.append("TẢI PHÂN BỐ TAM GIÁC")
    if data.uvls:
        for i, (q, a, b) in enumerate(data.uvls, 1):
            lines.append(f"UVL{i}: qmax = {q:.2f} kN/m, từ x = {a:.2f} đến x = {b:.2f} m")
    else:
        lines.append("Không có")

    # --- Moment tập trung ---
    lines.append("MOMENT TẬP TRUNG")
    if data.point_moments:
        for i, (M, x) in enumerate(data.point_moments, 1):
            lines.append(f"M{i} = {M:.2f} kNm tại x = {x:.2f} m")
    else:
        lines.append("Không có")

    return lines


def build_reaction_report_fixed(
        data: BeamInput, r1: float, r2: float,
        sum_p_mom: float, sum_udl_mom: float, sum_uvl_mom: float, sum_m_mom: float
) -> list[str]:
    lines = []
    lines.append("II. TÍNH TOÁN PHẢN LỰC GỐI (DẦM TỰA ĐƠN)")
    L = data.length

    # 1. Tính tổng mômen chi tiết
    lines.append("1) Lấy mômen tại gối A (x=0) để tìm phản lực R2:")
    lines.append("   ⇔ ΣM_A = 0 ⇔ R2*L + ΣM_tải = 0")
    lines.append("   Phân tích ΣM_tải:")

    # Chi tiết tải tập trung
    for P, x in data.point_loads:
        lines.append(f"   - Lực tập trung {P:+.2f} kN tại x={x:.2f}m: M = {P:+.2f} * {x:.2f} = {P * x:+.2f} kNm")

    # Chi tiết tải phân bố đều
    for q, a, b in data.udls:
        F = q * (b - a)
        x_center = a + (b - a) / 2
        lines.append(f"   - Tải đều {q:+.2f} kN/m từ {a:.2f}m đến {b:.2f}m:")
        lines.append(f"     ⇒ Lực tập trung F = {q:.2f} * {(b - a):.2f} = {F:.2f} kN tại x={x_center:.2f}m")
        lines.append(f"     ⇒ M = {F:.2f} * {x_center:.2f} = {F * x_center:+.2f} kNm")

    # Chi tiết tải tam giác
    for qmax, a, b in data.uvls:
        F = 0.5 * qmax * (b - a)
        # Nếu là dạng tăng dần (từ a đến b): trọng tâm cách a là 2/3 chiều dài
        x_center = a + (2 / 3 if data.uvl_type == "increase" else 1 / 3) * (b - a)
        lines.append(f"   - Tải tam giác {qmax:+.2f} kN/m từ {a:.2f}m đến {b:.2f}m:")
        lines.append(f"     ⇒ Lực tập trung F = 0.5 * {qmax:.2f} * {(b - a):.2f} = {F:.2f} kN tại x={x_center:.2f}m")
        lines.append(f"     ⇒ M = {F:.2f} * {x_center:.2f} = {F * x_center:+.2f} kNm")

    # Chi tiết moment tập trung
    for M, x in data.point_moments:
        lines.append(f"   - Moment tập trung {M:+.2f} kNm tại x={x:.2f}m: M = {M:+.2f} kNm")

    lines.append(f"   ⇒ Tổng ΣM_tải = {sum_p_mom + sum_udl_mom + sum_uvl_mom + sum_m_mom:+.2f} kNm")
    lines.append(f"   ⇒ R2 = -({sum_p_mom + sum_udl_mom + sum_uvl_mom + sum_m_mom:+.2f}) / {L:.2f} = {r2:.4f} kN")
    lines.append("")

    # 2. Chi tiết tính R1
    lines.append("2) Chiếu lực lên trục y để tìm R1:")
    lines.append("   ⇔ R1 + R2 + ΣF_đứng = 0 ⇒ R1 = -(ΣF_đứng + R2)")

    t_p = sum(P for P, _ in data.point_loads)
    t_udl = sum(q * (b - a) for q, a, b in data.udls)
    t_uvl = sum(0.5 * qmax * (b - a) for qmax, a, b in data.uvls)
    lines.append(f"   ⇒ ΣF_đứng = ({t_p:+.2f}) + ({t_udl:+.2f}) + ({t_uvl:+.2f}) = {t_p + t_udl + t_uvl:+.2f} kN")
    lines.append(f"   ⇒ R1 = -({t_p + t_udl + t_uvl:+.2f} + {r2:.4f}) = {r1:.4f} kN")

    return lines


def build_fixed_report_fixed(data: BeamInput, rv_fixed: float, mr_fixed: float) -> list[str]:
    lines = []
    lines.append("II. TÍNH TOÁN PHẢN LỰC NGÀM (DẦM CONSOLE)")
    L = data.length

    # 1. Tính phản lực đứng RV (ΣFy = 0)
    lines.append("1) Chiếu lực lên trục y để tìm phản lực đứng RV tại ngàm (x=L):")
    lines.append("   ⇔ RV + ΣF_đứng = 0 ⇒ RV = -ΣF_đứng")

    t_p = sum(P for P, _ in data.point_loads)
    t_udl = sum(q * (b - a) for q, a, b in data.udls)
    t_uvl = sum(0.5 * qmax * (b - a) for qmax, a, b in data.uvls)

    lines.append(f"   ⇒ ΣF_đứng = ({t_p:+.2f}) + ({t_udl:+.2f}) + ({t_uvl:+.2f}) = {t_p + t_udl + t_uvl:+.2f} kN")
    lines.append(f"   ⇒ RV = -({t_p + t_udl + t_uvl:+.2f}) = {rv_fixed:.4f} kN")
    lines.append("")

    # 2. Tính mômen ngàm MR (ΣM_ngàm = 0)
    lines.append(f"2) Cân bằng mômen tại ngàm (x={L:.2f}) để tìm mômen phản lực MR:")
    lines.append("   ⇔ MR + ΣM_tải_đối_với_ngàm = 0 ⇒ MR = -ΣM_tải_đối_với_ngàm")
    lines.append("   Phân tích ΣM_tải_đối_với_ngàm (cánh tay đòn tính từ vị trí đặt lực đến ngàm x=L):")

    sum_moments = 0.0

    # Chi tiết tải tập trung
    for P, x in data.point_loads:
        d = L - x
        m = P * d
        sum_moments += m
        lines.append(f"   - Lực tập trung {P:+.2f} kN tại x={x:.2f}m: M = {P:+.2f} * {d:.2f} = {m:+.2f} kNm")

    # Chi tiết tải phân bố đều
    for q, a, b in data.udls:
        F = q * (b - a)
        x_center = a + (b - a) / 2
        d = L - x_center
        m = F * d
        sum_moments += m
        lines.append(f"   - Tải đều {q:+.2f} kN/m từ {a:.2f}m đến {b:.2f}m:")
        lines.append(f"     ⇒ F = {q:.2f} * {(b - a):.2f} = {F:.2f} kN tại x={x_center:.2f}m")
        lines.append(f"     ⇒ M = {F:.2f} * {d:.2f} = {m:+.2f} kNm")

    # Chi tiết tải tam giác
    for qmax, a, b in data.uvls:
        F = 0.5 * qmax * (b - a)
        x_center = a + (2 / 3 if data.uvl_type == "increase" else 1 / 3) * (b - a)
        d = L - x_center
        m = F * d
        sum_moments += m
        lines.append(f"   - Tải tam giác {qmax:+.2f} kN/m từ {a:.2f}m đến {b:.2f}m:")
        lines.append(f"     ⇒ F = 0.5 * {qmax:.2f} * {(b - a):.2f} = {F:.2f} kN tại x={x_center:.2f}m")
        lines.append(f"     ⇒ M = {F:.2f} * {d:.2f} = {m:+.2f} kNm")
    # Chi tiết moment tập trung
    for M, x in data.point_moments:
        sum_moments += M
        lines.append(f"   - Moment tập trung {M:+.2f} kNm tại x={x:.2f}m: M = {M:+.2f} kNm")
    lines.append(f"   ⇒ Tổng ΣM_tải_đối_với_ngàm = {sum_moments:+.2f} kNm")
    lines.append(f"   ⇒ MR = -({sum_moments:+.2f}) = {mr_fixed:.4f} kNm")
    return lines
def build_shear_report(data: BeamInput, x: np.ndarray, shear: np.ndarray) -> list[str]:
    lines = []
    lines.append("III. THUYẾT MINH BIỂU ĐỒ LỰC CẮT (V)")
    lines.append("1) Giá trị lực cắt tại các điểm biên và vị trí có tải trọng:")

    critical_points = [0.0, data.length]
    if data.point_loads: critical_points.extend([pos for _, pos in data.point_loads])
    if data.udls:
        for _, a, b in data.udls: critical_points.extend([a, b])
    if data.uvls:
        for _, a, b in data.uvls: critical_points.extend([a, b])

    critical_points = sorted(list(set(critical_points)))

    for pt in critical_points:
        idx = np.argmin(np.abs(x - pt))
        val = shear[idx]
        lines.append(f"   - Tại x = {pt:.2f} m: V = {val:.2f} kN")

    idx_max = int(np.argmax(np.abs(shear)))
    lines.append(f"2) Cực trị: Lực cắt lớn nhất |Vmax| = {abs(shear[idx_max]):.2f} kN tại x = {x[idx_max]:.2f} m")
    return lines


def build_moment_report(data: BeamInput, x: np.ndarray, moment: np.ndarray, shear: np.ndarray) -> list[str]:
    lines = []
    lines.append("IV. THUYẾT MINH BIỂU ĐỒ MOMENT UỐN (M)")
    lines.append("1) Giá trị mômen tại các điểm biên và vị trí tập trung:")

    critical_points = [0.0, data.length]
    if data.point_moments: critical_points.extend([pos for _, pos in data.point_moments])
    if data.point_loads: critical_points.extend([pos for _, pos in data.point_loads])
    critical_points = sorted(list(set(critical_points)))

    for pt in critical_points:
        idx = np.argmin(np.abs(x - pt))
        lines.append(f"   - Tại x = {pt:.2f} m: M = {moment[idx]:.2f} kNm")

    zero_crossings = []
    for i in range(1, len(x) - 1):
        if (shear[i - 1] >= 0 and shear[i + 1] <= 0) or (shear[i - 1] <= 0 and shear[i + 1] >= 0):
            if not any(np.abs(x[i] - p) < 0.05 for p in critical_points):
                zero_crossings.append(i)

    if zero_crossings:
        lines.append("2) Phân tích cực trị hình học (Vị trí V = 0):")
        for idx in zero_crossings:
            lines.append(f"   - Tại đỉnh parabol x = {x[idx]:.2f} m ⇒ M = {moment[idx]:.2f} kNm")

    idx_max = int(np.argmax(np.abs(moment)))
    lines.append(f"⇒ Kết luận: Mômen uốn lớn nhất |Mmax| = {abs(moment[idx_max]):.2f} kNm tại x = {x[idx_max]:.2f} m")
    return lines


def build_deflection_report(
        data: BeamInput, x: np.ndarray, deflection: np.ndarray,
        c1: float, c2: float, w_int_L: float, theta_int_L: float
) -> list[str]:
    lines = []
    lines.append("V. ĐƯỜNG ĐÀN HỒI VÀ CHUYỂN VỊ")
    lines.append("1) Thiết lập hàm vi phân cơ bản:")
    lines.append("   - Hàm góc xoay: EI * θ(x) = ∫ -M(x)dx + C1")
    lines.append("   - Hàm độ võng:  EI * w(x) = ∫ [EI * θ(x)]dx + C2 = [Tích phân tải trọng] + C1*x + C2")
    lines.append("")

    is_simple = data.beam_type == "simple"
    L = float(data.length)

    lines.append("2) Xác định hằng số tích phân C1, C2 từ điều kiện biên:")
    if is_simple:
        lines.append("   - Điều kiện biên 1: Tại gối trái (x = 0) có độ võng w = 0")
        lines.append("     ⇔ [Tích phân tải tại 0] + C1*0 + C2 = 0")
        lines.append("     ⇒ C2 = 0.0000")
        lines.append(f"   - Điều kiện biên 2: Tại gối phải (x = {L:.2f}) có độ võng w = 0")
        lines.append("     ⇔ [Tích phân tải tại L] + C1*L + C2 = 0")
        lines.append(f"     ⇔ {w_int_L:.4f} + C1*{L:.2f} + 0 = 0")
        lines.append(f"     ⇒ C1 = {c1:.6f}")
    else:
        lines.append(f"   - Điều kiện biên 1: Tại ngàm cứng (x = {L:.2f}) có góc xoay θ = 0")
        lines.append("     ⇔ [Tích phân góc xoay tại L] + C1 = 0")
        lines.append(f"     ⇔ {theta_int_L:.4f} + C1 = 0")
        lines.append(f"     ⇒ C1 = {c1:.6f}")
        lines.append(f"   - Điều kiện biên 2: Tại ngàm cứng (x = {L:.2f}) có độ võng w = 0")
        lines.append("     ⇔ [Tích phân tải tại L] + C1*L + C2 = 0")
        lines.append(f"     ⇔ {w_int_L:.4f} + ({c1:.6f})*{L:.2f} + C2 = 0")
        lines.append(f"     ⇒ C2 = {c2:.6f}")
    lines.append("")
    idx_w = int(np.argmax(np.abs(deflection)))
    lines.append(f"3) Kết luận chuyển vị max: |w_max| = {abs(deflection[idx_w]):.6f}/EI tại x = {x[idx_w]:.2f} m")
    return lines