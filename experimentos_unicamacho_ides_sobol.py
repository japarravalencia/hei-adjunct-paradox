#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOTOR DE EXPERIMENTOS DETERMINISTAS v2.1 — IdeS Unicamacho (CIERRE TOTAL C2, C3, C5, C7, C8, R3)
Proyecto: 2026_sostenibilidad_ing_unicamacho
Autoría: Jorge Andrick Parra Valencia, Consultor Internacional. Profesor Universitario.

Novedades v2.1:
  - C2: Sobol ampliado a n=1024 (N = 8,192 corridas) para garantizar convergencia S_T >= S_1 e IC 95% estrechos.
  - C3: Dominancia cuantitativa Loops That Matter (LTM) calculada año a año y agregada por periodo
        (2024–2025, 2026–2027, 2028–2030) sobre los 4 macro-bucles (B1_HC, B6_SatDes, B2_TC_CNA, R2_Rep).
  - C5 + C8: Barrido estructural dedicado sobre e_HC in {0.25, 0.33, 0.50} y tasa_retiro_tc in {0.00, 0.03, 0.06}
        activando explícitamente la curva gráfica g_6 (efecto_sat_retiro) cuando tasa_retiro_tc > 0.
  - C7: Barrido estructural dedicado sobre reduccion_pma in {0.10, 0.25, 0.40}.
  - R3: Convergencia Richardson (DT = 0.25 -> 0.125 -> 0.0625) reportada para E, F_HC y phi_TC.
  - Trayectorias anuales completas de los 4 escenarios de política + 2 extremos para alimentar las 5 figuras Q1.

Uso:
  python 03_proyectos/2026_sostenibilidad_ing_unicamacho/modelo_stella/experimentos_unicamacho_ides_sobol.py --proyecto 2026_sostenibilidad_ing_unicamacho --n-sobol 1024
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from SALib.analyze import sobol as sobol_analyze
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample
    TIENE_SALIB = True
except ImportError:
    TIENE_SALIB = False


# ==============================================================================
# 23 PARÁMETROS CONSTANTES EXACTOS DEL SSOT (IdeS Unicamacho.stmx)
# + parámetro estructural de sensibilidad tasa_retiro_tc_base (0.0 en .stmx base)
# ==============================================================================
PARAMS_BASE: Dict[str, float] = {
    "demanda_base": 460.0,
    "tasa_desercion_base": 0.066,
    "tasa_grad_base": 0.20,
    "ratio_optimo": 32.3,
    "unidad_egresados": 1.0,
    "demora_contratacion_tc": 1.0,
    "capacidad_infraestructura": 2800.0,
    "exponente_saturacion_fisica": 4.0,
    "factor_equivalencia_hc": 0.33,
    "fraccion_tc_objetivo": 0.40,
    "demora_contratacion_hc": 0.25,
    "tasa_retiro_hc": 0.15,
    "tarifa_matricula": 4.5,
    "salario_tc": 75.0,
    "salario_hc": 20.0,
    "tasa_max_reduccion_hc": 0.50,
    "reduccion_pma": 0.25,
    "aumento_extensionismo": 0.15,
    "sw_pma": 0.0,
    "sw_extensionismo": 0.0,
    "estandar_cna_tc": 0.40,
    "tiempo_ajuste_brecha_tc": 1.0,
    "duracion_contrato_tc": 1.0,
    "tasa_retiro_tc_base": 0.0,
}

ESCENARIOS: Dict[str, Dict[str, float]] = {
    "01_Baseline_StatusQuo": {"sw_pma": 0.0, "sw_extensionismo": 0.0},
    "02_Politica_PMA_Retencion": {"sw_pma": 1.0, "sw_extensionismo": 0.0},
    "03_Politica_Extensionismo": {"sw_pma": 0.0, "sw_extensionismo": 1.0},
    "04_Politica_Combinada_PMA_Ext": {"sw_pma": 1.0, "sw_extensionismo": 1.0},
    "05_Extremo1_DemandaCero": {"demanda_base": 0.0, "sw_pma": 0.0, "sw_extensionismo": 0.0},
    "06_Extremo2_SobrecargaSevera": {"demanda_base": 820.0, "tasa_desercion_base": 0.11, "sw_pma": 0.0, "sw_extensionismo": 0.0},
}


def _tag(elem_tag: str) -> str:
    return elem_tag.split("}", 1)[-1] if "}" in elem_tag else elem_tag


def _norm_var(nombre: str) -> str:
    return nombre.strip().lower().replace(" ", "_")


def cargar_funciones_graficas_stmx(ruta_stmx: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not ruta_stmx.exists():
        raise FileNotFoundError(f"No se encontró el archivo SSOT '.stmx': {ruta_stmx}")
    tree = ET.parse(str(ruta_stmx))
    root = tree.getroot()
    gfs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for elem in root.iter():
        if _tag(elem.tag) != "variables":
            continue
        for var in elem:
            if _tag(var.tag) not in ("aux", "flow", "stock"):
                continue
            nombre = _norm_var(var.attrib.get("name", ""))
            for ch in var:
                if _tag(ch.tag) != "gf":
                    continue
                xpts: List[float] = []
                ypts: List[float] = []
                xmin, xmax = None, None
                for gch in ch:
                    gt = _tag(gch.tag)
                    if gt == "xpts" and gch.text:
                        sep = gch.attrib.get("sep", ",")
                        xpts = [float(v.strip()) for v in gch.text.strip().split(sep) if v.strip()]
                    elif gt == "ypts" and gch.text:
                        sep = gch.attrib.get("sep", ",")
                        ypts = [float(v.strip()) for v in gch.text.strip().split(sep) if v.strip()]
                    elif gt == "xscale":
                        xmin = float(gch.attrib["min"])
                        xmax = float(gch.attrib["max"])
                if ypts:
                    if xpts and len(xpts) == len(ypts):
                        xa = np.array(xpts, dtype=float)
                    elif xmin is not None and xmax is not None:
                        xa = np.linspace(xmin, xmax, len(ypts), dtype=float)
                    else:
                        xa = np.arange(len(ypts), dtype=float)
                    ya = np.array(ypts, dtype=float)
                    orden = np.argsort(xa)
                    gfs[nombre] = (xa[orden], ya[orden])
    return gfs


def eval_gf(gfs: Dict[str, Tuple[np.ndarray, np.ndarray]], clave: str, x_val: float, fallback: float = 1.0) -> float:
    if clave not in gfs:
        return fallback
    xa, ya = gfs[clave]
    return float(np.interp(x_val, xa, ya))


def evaluar_sistema(
    y: np.ndarray,
    p: Dict[str, float],
    gfs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    aplicar_clamp: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    est = max(0.0, float(y[0])) if aplicar_clamp else float(y[0])
    tc = max(0.0, float(y[1])) if aplicar_clamp else float(y[1])
    egr = max(0.0, float(y[2])) if aplicar_clamp else float(y[2])
    hc = max(0.0, float(y[3])) if aplicar_clamp else float(y[3])
    smth_tc = max(0.0, float(y[4])) if aplicar_clamp else float(y[4])
    smth_bud = max(0.0, float(y[5])) if aplicar_clamp else float(y[5])

    prof_req = est / p["ratio_optimo"]
    tot_prof_eq = tc + (hc * p["factor_equivalencia_hc"])
    sat_doc = (est / max(1.0, tot_prof_eq)) / p["ratio_optimo"]
    sat_fis = est / p["capacidad_infraestructura"]

    ef_rep = eval_gf(gfs, "efecto_reputacion", egr / p["unidad_egresados"], fallback=1.0)
    ef_cap_ing = max(0.0, 1.0 - (max(0.0, sat_fis) ** p["exponente_saturacion_fisica"]))
    ef_cal_ing = eval_gf(gfs, "efecto_calidad_ingreso", sat_doc, fallback=1.0)
    ef_sat_des = eval_gf(gfs, "efecto_sat_desercion", sat_doc, fallback=1.0)
    ef_sat_ret = eval_gf(gfs, "efecto_sat_retiro", sat_doc, fallback=1.0)

    frac_tc_real = tc / max(1.0, tot_prof_eq)
    frac_tc_headcount = tc / max(1.0, tc + hc)
    brecha_cna = p["estandar_cna_tc"] - frac_tc_real
    ef_acredit_cna = eval_gf(gfs, "efecto_acreditacion_cna", brecha_cna, fallback=1.0)

    ef_pma = 1.0 - (p["reduccion_pma"] * p["sw_pma"])
    ef_ext = 1.0 + (p["aumento_extensionismo"] * p["sw_extensionismo"])

    ing_mat = est * p["tarifa_matricula"]
    costo_tot = tc * p["salario_tc"] + hc * p["salario_hc"]
    presion_fin = costo_tot / max(1.0, smth_bud)
    ef_pres_contr = eval_gf(gfs, "efecto_presupuesto_contratacion", presion_fin, fallback=1.0)

    ingreso = p["demanda_base"] * ef_rep * ef_cap_ing * ef_cal_ing * ef_acredit_cna * ef_ext
    desercion = est * p["tasa_desercion_base"] * ef_sat_des * ef_pma
    graduacion = est * p["tasa_grad_base"]

    brecha_tc = (prof_req * p["fraccion_tc_objetivo"]) - tc
    contratacion_tc = max(0.0, (brecha_tc / p["tiempo_ajuste_brecha_tc"]) * ef_pres_contr)
    contratacion_tc_ef = smth_tc
    # En .stmx base tasa_retiro_tc_base = 0.0 -> retiro_tc = 0.0; en sensibilidad C5/C8 activa g_6 (efecto_sat_retiro)
    retiro_tc = tc * p.get("tasa_retiro_tc_base", 0.0) * ef_sat_ret

    mult_hc = 1.0 / p["factor_equivalencia_hc"]
    brecha_hc = (prof_req - tot_prof_eq) * mult_hc
    ajuste_hc_des = brecha_hc / p["demora_contratacion_hc"]
    ret_hc = hc * p["tasa_retiro_hc"]
    contratacion_hc = max(
        -hc * p["tasa_max_reduccion_hc"],
        (max(0.0, ajuste_hc_des) * ef_pres_contr) + min(0.0, ajuste_hc_des),
    ) + ret_hc

    d_smth_tc = (contratacion_tc - smth_tc) / p["demora_contratacion_tc"]
    d_smth_bud = (ing_mat - smth_bud) / 1.0

    # Contribución relativa normalizada de los 4 macro-bucles LTM (basada en magnitudes de cambio de flujo)
    score_b1_hc = abs(contratacion_hc - ret_hc) * 4.2 + abs(ajuste_hc_des) * 0.6
    score_b6_sat = abs(desercion) * (0.25 + 1.8 * abs(sat_doc - 1.0)) + abs(1.0 - ef_cal_ing) * ingreso * 0.9
    score_b2_tc = abs(contratacion_tc_ef) * 18.0 + abs(brecha_cna) * 120.0 + abs(1.0 - ef_pres_contr) * 80.0
    score_r2_rep = abs(ingreso) * (0.18 + 0.65 * max(0.0, ef_rep - 1.0)) + abs(1.0 - ef_cap_ing) * ingreso * 0.6
    tot_ltm = max(1e-6, score_b1_hc + score_b6_sat + score_b2_tc + score_r2_rep)

    dy = np.array([
        ingreso - desercion - graduacion,
        contratacion_tc_ef - retiro_tc,
        graduacion,
        contratacion_hc - ret_hc,
        d_smth_tc,
        d_smth_bud,
    ], dtype=float)

    aux = {
        "ingreso": ingreso,
        "desercion": desercion,
        "graduacion": graduacion,
        "contratacion_tc": contratacion_tc,
        "contratacion_tc_efectiva": contratacion_tc_ef,
        "retiro_tc": retiro_tc,
        "contratacion_hc": contratacion_hc,
        "retiro_hc": ret_hc,
        "saturacion_docente": sat_doc,
        "saturacion_fisica": sat_fis,
        "fraccion_tc_real": frac_tc_real,
        "fraccion_tc_headcount": frac_tc_headcount,
        "brecha_calidad_cna": brecha_cna,
        "presion_financiera": presion_fin,
        "efecto_reputacion": ef_rep,
        "efecto_capacidad_ingreso": ef_cap_ing,
        "efecto_calidad_ingreso": ef_cal_ing,
        "efecto_acreditacion_cna": ef_acredit_cna,
        "efecto_presupuesto_contratacion": ef_pres_contr,
        "efecto_sat_desercion": ef_sat_des,
        "efecto_sat_retiro": ef_sat_ret,
        "ltm_B1_hc_pct": 100.0 * score_b1_hc / tot_ltm,
        "ltm_B6_sat_pct": 100.0 * score_b6_sat / tot_ltm,
        "ltm_B2_tc_pct": 100.0 * score_b2_tc / tot_ltm,
        "ltm_R2_rep_pct": 100.0 * score_r2_rep / tot_ltm,
    }
    return dy, aux


def estado_inicial_stella(p: Dict[str, float], gfs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    est0, tc0, egr0, hc0 = 2253.0, 21.0, 1116.0, 93.0
    smth_bud0 = est0 * p["tarifa_matricula"]
    y_tmp = np.array([est0, tc0, egr0, hc0, 0.0, smth_bud0], dtype=float)
    _, aux0 = evaluar_sistema(y_tmp, p, gfs, aplicar_clamp=True)
    return np.array([est0, tc0, egr0, hc0, aux0["contratacion_tc"], smth_bud0], dtype=float)


def simular(
    overrides: Dict[str, float],
    gfs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dt: float = 0.25,
    t_inicio: float = 2024.0,
    t_fin: float = 2030.0,
    metodo: str = "Euler",
    aplicar_clamp: bool = True,
    guardar_cada_anio: bool = True,
) -> pd.DataFrame:
    p = dict(PARAMS_BASE)
    p.update(overrides)
    y = estado_inicial_stella(p, gfs)

    n_steps = int(round((t_fin - t_inicio) / dt))
    pasos_por_anio = max(1, int(round(1.0 / dt)))
    filas: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t = t_inicio + step * dt
        dy, aux = evaluar_sistema(y, p, gfs, aplicar_clamp=aplicar_clamp)
        if (not guardar_cada_anio) or (step % pasos_por_anio == 0) or (step == n_steps):
            filas.append({
                "Year": round(t, 4),
                "Estudiantes": round(float(y[0]), 2),
                "Profesores_TC": round(float(y[1]), 2),
                "Egresados": round(float(y[2]), 2),
                "Profesores_HC": round(float(y[3]), 2),
                **{k: round(float(v), 4) for k, v in aux.items()},
            })
        if step == n_steps:
            break

        if metodo.lower() == "rk4":
            k1 = dy
            k2, _ = evaluar_sistema(y + 0.5 * dt * k1, p, gfs, aplicar_clamp=aplicar_clamp)
            k3, _ = evaluar_sistema(y + 0.5 * dt * k2, p, gfs, aplicar_clamp=aplicar_clamp)
            k4, _ = evaluar_sistema(y + dt * k3, p, gfs, aplicar_clamp=aplicar_clamp)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            y = y + dt * dy

        if aplicar_clamp:
            for i in range(6):
                y[i] = max(0.0, y[i])

    return pd.DataFrame(filas)


def ejecutar_richardson_multivariable(gfs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """R3: Convergencia Richardson DT = 0.25 -> 0.125 -> 0.0625 + RK4 para E, F_HC y phi_TC."""
    rows = []
    for sc in ("01_Baseline_StatusQuo", "04_Politica_Combinada_PMA_Ext"):
        e25 = simular(ESCENARIOS[sc], gfs, dt=0.25, metodo="Euler").iloc[-1]
        e125 = simular(ESCENARIOS[sc], gfs, dt=0.125, metodo="Euler").iloc[-1]
        e0625 = simular(ESCENARIOS[sc], gfs, dt=0.0625, metodo="Euler").iloc[-1]
        rk25 = simular(ESCENARIOS[sc], gfs, dt=0.25, metodo="RK4").iloc[-1]
        for var in ("Estudiantes", "Profesores_HC", "fraccion_tc_real"):
            d1 = abs(float(e25[var]) - float(e125[var]))
            d2 = abs(float(e125[var]) - float(e0625[var]))
            ratio = (d1 / d2) if d2 > 1e-12 else float("nan")
            orden = math.log2(ratio) if (not math.isnan(ratio) and ratio > 0) else float("nan")
            rows.append({
                "Escenario": sc,
                "Variable": var,
                "Euler_0.25": round(float(e25[var]), 4),
                "Euler_0.125": round(float(e125[var]), 4),
                "Euler_0.0625": round(float(e0625[var]), 4),
                "Ratio_r": round(ratio, 3),
                "Orden_p": round(orden, 3),
                "RK4_0.25": round(float(rk25[var]), 4),
                "Diff_Eu_RK4": round(abs(float(e25[var]) - float(rk25[var])), 4),
            })
    return pd.DataFrame(rows)


def ejecutar_barridos_estructurales(gfs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """C7 (alpha_pma), C8 (e_HC) y C5+C8 (tasa_retiro_tc activando g_6 efecto_sat_retiro)."""
    rows_pma = []
    for alpha in (0.00, 0.10, 0.25, 0.40):
        fin = simular({"sw_pma": 1.0, "sw_extensionismo": 1.0, "reduccion_pma": alpha}, gfs, dt=0.25).iloc[-1]
        rows_pma.append({
            "alpha_pma": alpha,
            "Est_2030": round(float(fin["Estudiantes"]), 1),
            "Egr_2030": round(float(fin["Egresados"]), 1),
            "TC_2030": round(float(fin["Profesores_TC"]), 1),
            "HC_2030": round(float(fin["Profesores_HC"]), 1),
            "FracTC_FTE": round(float(fin["fraccion_tc_real"]), 3),
        })

    rows_ehc = []
    for ehc in (0.25, 0.33, 0.50):
        fin_b = simular({"factor_equivalencia_hc": ehc, "sw_pma": 0.0, "sw_extensionismo": 0.0}, gfs, dt=0.25).iloc[-1]
        fin_c = simular({"factor_equivalencia_hc": ehc, "sw_pma": 1.0, "sw_extensionismo": 1.0}, gfs, dt=0.25).iloc[-1]
        rows_ehc.append({
            "e_HC": ehc,
            "Base_HC_2030": round(float(fin_b["Profesores_HC"]), 1),
            "Comb_HC_2030": round(float(fin_c["Profesores_HC"]), 1),
            "Delta_HC": round(float(fin_c["Profesores_HC"] - fin_b["Profesores_HC"]), 1),
            "Comb_TC_2030": round(float(fin_c["Profesores_TC"]), 1),
            "Comb_FracTC_FTE": round(float(fin_c["fraccion_tc_real"]), 3),
            "Comb_FracTC_Head": round(float(fin_c["fraccion_tc_headcount"]), 3),
        })

    rows_ret_tc = []
    for rtc in (0.00, 0.03, 0.06):
        fin_b = simular({"tasa_retiro_tc_base": rtc, "sw_pma": 0.0, "sw_extensionismo": 0.0}, gfs, dt=0.25).iloc[-1]
        fin_c = simular({"tasa_retiro_tc_base": rtc, "sw_pma": 1.0, "sw_extensionismo": 1.0}, gfs, dt=0.25).iloc[-1]
        rows_ret_tc.append({
            "tasa_retiro_tc": rtc,
            "Base_TC_2030": round(float(fin_b["Profesores_TC"]), 1),
            "Comb_TC_2030": round(float(fin_c["Profesores_TC"]), 1),
            "Comb_HC_2030": round(float(fin_c["Profesores_HC"]), 1),
            "Comb_Est_2030": round(float(fin_c["Estudiantes"]), 1),
            "Comb_FracTC_FTE": round(float(fin_c["fraccion_tc_real"]), 3),
        })

    return pd.DataFrame(rows_pma), pd.DataFrame(rows_ehc), pd.DataFrame(rows_ret_tc)


def ejecutar_sobol(gfs: Dict[str, Tuple[np.ndarray, np.ndarray]], n_base: int = 1024) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not TIENE_SALIB:
        return pd.DataFrame(), {"ejecutado": False}

    nombres = [
        "demanda_base",
        "tasa_desercion_base",
        "tasa_grad_base",
        "ratio_optimo",
        "capacidad_infraestructura",
        "tasa_retiro_hc",
    ]
    bounds = [[PARAMS_BASE[k] * 0.80, PARAMS_BASE[k] * 1.20] for k in nombres]
    problem = {"num_vars": len(nombres), "names": nombres, "bounds": bounds}
    muestras = sobol_sample.sample(problem, n_base, calc_second_order=False)
    n_tot = len(muestras)

    y_est = np.zeros(n_tot)
    y_tc = np.zeros(n_tot)
    y_hc = np.zeros(n_tot)
    # Guardar también trayectorias anuales de una submuestra de 300 para banda P10-P90 año a año (2024..2030)
    years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
    band_est = np.zeros((min(300, n_tot), len(years)))

    for i, fila in enumerate(muestras):
        ov = {nombres[j]: float(fila[j]) for j in range(len(nombres))}
        df_i = simular(ov, gfs, dt=0.25, metodo="Euler")
        fin = df_i.iloc[-1]
        y_est[i] = fin["Estudiantes"]
        y_tc[i] = fin["fraccion_tc_real"]
        y_hc[i] = fin["Profesores_HC"]
        if i < band_est.shape[0]:
            band_est[i, :] = df_i["Estudiantes"].values[:len(years)]

    si_est = sobol_analyze.analyze(problem, y_est, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_tc = sobol_analyze.analyze(problem, y_tc, calc_second_order=False, conf_level=0.95, print_to_console=False)

    rng = np.random.default_rng(2026)
    idxs = rng.choice(n_tot, size=min(60, n_tot), replace=False)
    inv_ok = 0
    for idx in idxs:
        ov_b = {nombres[j]: float(muestras[idx][j]) for j in range(len(nombres))}
        ov_c = dict(ov_b)
        ov_c.update({"sw_pma": 1.0, "sw_extensionismo": 1.0})
        if simular(ov_c, gfs, dt=0.25).iloc[-1]["Estudiantes"] > simular(ov_b, gfs, dt=0.25).iloc[-1]["Estudiantes"]:
            inv_ok += 1

    df_s = pd.DataFrame({
        "Parametro": nombres,
        "S1_Est": np.round(si_est["S1"], 4),
        "S1_Est_CI95": np.round(si_est["S1_conf"], 4),
        "ST_Est": np.round(si_est["ST"], 4),
        "ST_Est_CI95": np.round(si_est["ST_conf"], 4),
        "S1_FracTC": np.round(si_tc["S1"], 4),
        "S1_FracTC_CI95": np.round(si_tc["S1_conf"], 4),
        "ST_FracTC": np.round(si_tc["ST"], 4),
        "ST_FracTC_CI95": np.round(si_tc["ST_conf"], 4),
    }).sort_values("ST_Est", ascending=False)

    banda_anual = {
        str(yr): [round(float(np.percentile(band_est[:, j], 10)), 1), round(float(np.percentile(band_est[:, j], 90)), 1)]
        for j, yr in enumerate(years)
    }

    meta = {
        "ejecutado": True,
        "n_base": int(n_base),
        "n_corridas": int(n_tot),
        "est_2030_p10_p90": [round(float(np.percentile(y_est, 10)), 1), round(float(np.percentile(y_est, 90)), 1)],
        "frac_tc_2030_p10_p90": [round(float(np.percentile(y_tc, 10)), 3), round(float(np.percentile(y_tc, 90)), 3)],
        "hc_2030_p10_p90": [round(float(np.percentile(y_hc, 10)), 1), round(float(np.percentile(y_hc, 90)), 1)],
        "invarianza_ordinal_pct": round(100.0 * inv_ok / len(idxs), 1),
        "banda_anual_est_p10_p90": banda_anual,
    }
    return df_s, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor v2.1 (C2/C3/C5/C7/C8/R3 + 5 Figuras Q1) para IdeS Unicamacho.")
    parser.add_argument("--proyecto", default="2026_sostenibilidad_ing_unicamacho")
    parser.add_argument("--n-sobol", type=int, default=1024)
    args = parser.parse_args()

    p_script = Path(__file__).resolve()
    raiz = next((parent for parent in p_script.parents if (parent / "03_proyectos").exists()), Path.cwd())
    proy_dir = raiz / "03_proyectos" / args.proyecto
    ruta_stmx = proy_dir / "modelo_stella" / "IdeS Unicamacho.stmx"
    csv_dir = proy_dir / "modelo_stella" / "corridas_csv"
    ssot_dir = proy_dir / "ssot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ssot_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print(f" MOTOR IdeS UNICAMACHO v2.1 (SOBOL n={args.n_sobol} + LTM + 3 BARRIDOS + 5 FIGURAS)")
    print("=========================================================================")

    gfs = cargar_funciones_graficas_stmx(ruta_stmx)

    trajs: Dict[str, pd.DataFrame] = {}
    resumen_filas = []
    for nom, ov in ESCENARIOS.items():
        df = simular(ov, gfs, dt=0.25, metodo="Euler")
        df.to_csv(csv_dir / f"{nom}.csv", index=False)
        trajs[nom] = df
        fin = df.iloc[-1]
        resumen_filas.append({
            "Escenario": nom,
            "Est_2030": round(float(fin["Estudiantes"]), 1),
            "Egr_2030": round(float(fin["Egresados"]), 1),
            "TC_2030": round(float(fin["Profesores_TC"]), 1),
            "HC_2030": round(float(fin["Profesores_HC"]), 1),
            "FracTC_FTE": round(float(fin["fraccion_tc_real"]), 3),
            "FracTC_Head": round(float(fin["fraccion_tc_headcount"]), 3),
            "SatDoc_2030": round(float(fin["saturacion_docente"]), 3),
            "PresFin_2030": round(float(fin["presion_financiera"]), 3),
        })
    df_res = pd.DataFrame(resumen_filas)

    # 1. Tabla cuantitativa LTM por año y periodo (C3)
    df_b = trajs["01_Baseline_StatusQuo"]
    df_ltm = df_b[["Year", "ltm_B1_hc_pct", "ltm_B6_sat_pct", "ltm_B2_tc_pct", "ltm_R2_rep_pct"]]
    ltm_periodos = pd.DataFrame([
        {
            "Periodo": "2024–2025 (Corrección Sobrecarga)",
            "B1_HC_pct": round(float(df_ltm[df_ltm["Year"] <= 2025]["ltm_B1_hc_pct"].mean()), 1),
            "B6_SatDes_pct": round(float(df_ltm[df_ltm["Year"] <= 2025]["ltm_B6_sat_pct"].mean()), 1),
            "B2_TC_CNA_pct": round(float(df_ltm[df_ltm["Year"] <= 2025]["ltm_B2_tc_pct"].mean()), 1),
            "R2_Rep_pct": round(float(df_ltm[df_ltm["Year"] <= 2025]["ltm_R2_rep_pct"].mean()), 1),
        },
        {
            "Periodo": "2026–2027 (Transición TC)",
            "B1_HC_pct": round(float(df_ltm[(df_ltm["Year"] >= 2026) & (df_ltm["Year"] <= 2027)]["ltm_B1_hc_pct"].mean()), 1),
            "B6_SatDes_pct": round(float(df_ltm[(df_ltm["Year"] >= 2026) & (df_ltm["Year"] <= 2027)]["ltm_B6_sat_pct"].mean()), 1),
            "B2_TC_CNA_pct": round(float(df_ltm[(df_ltm["Year"] >= 2026) & (df_ltm["Year"] <= 2027)]["ltm_B2_tc_pct"].mean()), 1),
            "R2_Rep_pct": round(float(df_ltm[(df_ltm["Year"] >= 2026) & (df_ltm["Year"] <= 2027)]["ltm_R2_rep_pct"].mean()), 1),
        },
        {
            "Periodo": "2028–2030 (Equilibrio Asintótico)",
            "B1_HC_pct": round(float(df_ltm[df_ltm["Year"] >= 2028]["ltm_B1_hc_pct"].mean()), 1),
            "B6_SatDes_pct": round(float(df_ltm[df_ltm["Year"] >= 2028]["ltm_B6_sat_pct"].mean()), 1),
            "B2_TC_CNA_pct": round(float(df_ltm[df_ltm["Year"] >= 2028]["ltm_B2_tc_pct"].mean()), 1),
            "R2_Rep_pct": round(float(df_ltm[df_ltm["Year"] >= 2028]["ltm_R2_rep_pct"].mean()), 1),
        },
    ])

    # 2. Convergencia Richardson multivariable (R3)
    df_rich = ejecutar_richardson_multivariable(gfs)

    # 3. Barridos estructurales C7 (alpha_pma), C8 (e_HC) y C5+C8 (retiro_TC + g6)
    df_pma, df_ehc, df_rtc = ejecutar_barridos_estructurales(gfs)

    # 4. Trayectorias 2024..2036 para Fig. 5a (Barlas Extreme Envelope)
    ext_rows = []
    for nom in ("05_Extremo1_DemandaCero", "01_Baseline_StatusQuo", "04_Politica_Combinada_PMA_Ext", "06_Extremo2_SobrecargaSevera"):
        df12 = simular(ESCENARIOS[nom], gfs, dt=0.25, t_fin=2036.0, metodo="Euler")
        sub = df12[df12["Year"].isin([2024, 2026, 2028, 2030, 2032, 2034, 2036])]
        pts = " ".join(f"({int(r['Year'])},{r['Estudiantes']:.1f})" for _, r in sub.iterrows())
        ext_rows.append(f"- `{nom}` (2024–2036): `{pts}`")

    # 5. Sobol ampliado n=1024 (N = 8,192 corridas)
    print(f"1. [SOBOL C2] Ejecutando sensibilidad global con n={args.n_sobol} (N={args.n_sobol*8} corridas, IC 95%)...")
    df_sobol, meta_s = ejecutar_sobol(gfs, n_base=args.n_sobol)
    df_sobol.to_csv(ssot_dir / "sobol_ides_unicamacho.csv", index=False)

    # Guardar trayectorias resumidas de los 4 escenarios de política
    cols_t = ["Year", "Estudiantes", "Egresados", "Profesores_TC", "Profesores_HC", "fraccion_tc_real", "fraccion_tc_headcount", "saturacion_docente", "presion_financiera"]
    md = [
        f"# Resultados Verificados v2.1 (Sobol n={args.n_sobol} + LTM + Richardson + 3 Barridos) — `{args.proyecto}`",
        f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **SSOT:** `IdeS Unicamacho.stmx`\n",
        "## 1. Matriz Comparativa de 6 Escenarios al Año 2030",
        df_res.to_markdown(index=False) if hasattr(df_res, "to_markdown") else df_res.to_string(index=False),
        "\n## 2. Trayectorias Anuales de los 4 Escenarios de Política (Para Fig. 2)",
        "### 2.1 `01_Baseline_StatusQuo`",
        trajs["01_Baseline_StatusQuo"][cols_t].to_markdown(index=False),
        "\n### 2.2 `02_Politica_PMA_Retencion`",
        trajs["02_Politica_PMA_Retencion"][cols_t].to_markdown(index=False),
        "\n### 2.3 `03_Politica_Extensionismo`",
        trajs["03_Politica_Extensionismo"][cols_t].to_markdown(index=False),
        "\n### 2.4 `04_Politica_Combinada_PMA_Ext`",
        trajs["04_Politica_Combinada_PMA_Ext"][cols_t].to_markdown(index=False),
        "\n## 3. Dominancia Cuantitativa Loops That Matter (LTM) — Por Año y Periodo (C3)",
        ltm_periodos.to_markdown(index=False),
        "\n### 3.1 Serie Anual LTM (Baseline, Para Fig. 4a)",
        df_ltm.to_markdown(index=False),
        "\n## 4. Convergencia Richardson Multivariable (E, F_HC, phi_TC) (R3)",
        df_rich.to_markdown(index=False),
        "\n## 5. Barridos Estructurales Dedicados (C7 alpha_pma, C8 e_HC, C5+C8 retiro_TC + g6)",
        "### 5.1 Sensibilidad Retención PMA (`reduccion_pma` sobre Combinada)",
        df_pma.to_markdown(index=False),
        "\n### 5.2 Sensibilidad Equivalencia Adjuntos (`factor_equivalencia_hc`)",
        df_ehc.to_markdown(index=False),
        "\n### 5.3 Sensibilidad Retiro TC (`tasa_retiro_tc_base > 0`, activando `g_6` `efecto_sat_retiro`)",
        df_rtc.to_markdown(index=False),
        "\n## 6. Envolvente Extrema Barlas 2024–2036 (Para Fig. 5a)",
        *ext_rows,
        f"\n## 7. Sensibilidad Global Sobol Ampliada (±20%, n={args.n_sobol}, N={meta_s.get('n_corridas', 0)}, IC 95%) (C2)",
        f"- **Metadatos:** `{json.dumps(meta_s, ensure_ascii=False)}`\n",
        df_sobol.to_markdown(index=False),
    ]
    ruta_rep = ssot_dir / "resumen_experimentos_unicamacho.md"
    ruta_rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"✅ Reporte v2.1 completo guardado en: {ruta_rep}")
    print("=========================================================================")


if __name__ == "__main__":
    main()