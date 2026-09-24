from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_sortables import sort_items

try:
    import pgeocode
except ImportError:
    pgeocode = None


# ------------------------------------------------------------
# Grundeinstellungen
# ------------------------------------------------------------
st.set_page_config(
    page_title="Feiertags-Tourenplaner",
    page_icon="🧭",
    layout="wide",
)

SHEETS = ["DIREKT", "MK", "HUPA_NMS", "HUPA_MALCHOW"]
DAY_COLUMNS = ["Mo", "Die", "Mitt", "Don", "Fr", "Sam"]
DAY_LABELS = {
    "Mo": "Montag",
    "Die": "Dienstag",
    "Mitt": "Mittwoch",
    "Don": "Donnerstag",
    "Fr": "Freitag",
    "Sam": "Samstag",
}

DEFAULT_DEPOT_PLZ = {
    "DIREKT": "24539",
    "MK": "24539",
    "HUPA_NMS": "24539",
    "HUPA_MALCHOW": "17213",
}

PALETTE = [
    [157, 92, 255],
    [255, 154, 60],
    [111, 194, 118],
    [238, 99, 99],
    [207, 133, 255],
    [255, 196, 87],
    [132, 169, 255],
    [205, 113, 157],
    [144, 190, 178],
    [230, 145, 92],
]

st.markdown(
    """
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1700px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color:#9ca3af; font-size:.86rem;}
.route-pill {display:inline-block; padding:.18rem .55rem; border-radius:999px; background:#29252f; margin-right:.35rem;}
div[data-testid="stExpander"] {border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Hilfsfunktionen: Datei / Daten
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_source(file_bytes: bytes) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    missing = [s for s in SHEETS if s not in xl.sheet_names]
    if missing:
        raise ValueError("Diese Blätter fehlen: " + ", ".join(missing))

    for sheet in SHEETS:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
        df = df.rename(columns={"Straße": "Strasse"})
        expected = ["CSB", "SAP", "Name", "Strasse", "Plz", "Ort", *DAY_COLUMNS]
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected].copy()
        df["Quelle"] = sheet
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data["SAP"] = pd.to_numeric(all_data["SAP"], errors="coerce").astype("Int64")
    all_data["CSB"] = pd.to_numeric(all_data["CSB"], errors="coerce").astype("Int64")
    all_data["Plz"] = (
        pd.to_numeric(all_data["Plz"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(5)
    )
    for d in DAY_COLUMNS:
        all_data[d] = pd.to_numeric(all_data[d], errors="coerce").astype("Int64")

    all_data = all_data.dropna(subset=["SAP", "Name", "Plz"]).copy()
    all_data["uid"] = all_data.apply(
        lambda r: f"{r['Quelle']}::{int(r['SAP'])}::{int(r['CSB']) if pd.notna(r['CSB']) else 'x'}",
        axis=1,
    )
    return all_data


@st.cache_resource(show_spinner=False)
def get_postcode_geocoder():
    if pgeocode is None:
        return None
    return pgeocode.Nominatim("de")


@st.cache_data(show_spinner=False)
def geocode_postcodes(postcodes: Tuple[str, ...]) -> pd.DataFrame:
    geo = get_postcode_geocoder()
    if geo is None:
        return pd.DataFrame(columns=["Plz", "latitude", "longitude", "place_name"])

    unique = sorted({str(p).zfill(5) for p in postcodes if p and str(p) != "<NA>"})
    if not unique:
        return pd.DataFrame(columns=["Plz", "latitude", "longitude", "place_name"])

    result = geo.query_postal_code(unique)
    if isinstance(result, pd.Series):
        result = result.to_frame().T
    out = pd.DataFrame(
        {
            "Plz": pd.Series(unique, dtype="string"),
            "latitude": pd.to_numeric(result["latitude"], errors="coerce").to_numpy(),
            "longitude": pd.to_numeric(result["longitude"], errors="coerce").to_numpy(),
            "place_name": result.get("place_name", pd.Series([None] * len(unique))).to_numpy(),
        }
    )
    return out


def add_geo(df: pd.DataFrame) -> pd.DataFrame:
    coords = geocode_postcodes(tuple(df["Plz"].dropna().astype(str).unique().tolist()))
    return df.merge(coords, how="left", on="Plz")


def filter_customers(
    all_data: pd.DataFrame,
    days: List[str],
    sources: List[str],
) -> pd.DataFrame:
    df = all_data[all_data["Quelle"].isin(sources)].copy()
    if not days:
        return df.iloc[0:0].copy()

    mask = pd.Series(False, index=df.index)
    for d in days:
        mask |= df[d].notna()
    df = df[mask].copy()

    def old_tours(row) -> str:
        parts = []
        for d in days:
            if pd.notna(row[d]):
                parts.append(f"{DAY_LABELS[d][:2]} {int(row[d])}")
        return " / ".join(parts)

    df["Bisherige_Touren"] = df.apply(old_tours, axis=1)
    return df


# ------------------------------------------------------------
# Geo / Optimierung
# ------------------------------------------------------------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def projected_xy(df: pd.DataFrame) -> np.ndarray:
    lat = df["latitude"].astype(float).to_numpy()
    lon = df["longitude"].astype(float).to_numpy()
    mlat = np.deg2rad(np.nanmean(lat))
    return np.column_stack([lon * math.cos(mlat), lat])


def farthest_seeds(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.array([int(np.argmin(points[:, 0]))])
    center = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    seeds = [first]
    while len(seeds) < k:
        dist_to_seed = np.min(
            np.stack([np.linalg.norm(points - points[s], axis=1) for s in seeds], axis=1),
            axis=1,
        )
        for s in seeds:
            dist_to_seed[s] = -1
        seeds.append(int(np.argmax(dist_to_seed)))
    return np.array(seeds)


def balanced_kmeans(df: pd.DataFrame, max_size: int) -> np.ndarray:
    """Deterministische, kapazitätsbegrenzte Geo-Cluster ohne sklearn."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, math.ceil(n / max_size))
    if k == 1:
        return np.zeros(n, dtype=int)

    pts = projected_xy(df)
    seeds = farthest_seeds(pts, k)
    centers = pts[seeds].copy()

    base = n // k
    rem = n % k
    target = np.array([base + (1 if i < rem else 0) for i in range(k)], dtype=int)

    assignment = np.full(n, -1, dtype=int)
    for _ in range(25):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        sorted_d = np.sort(d, axis=1)
        regret = sorted_d[:, 1] - sorted_d[:, 0] if k > 1 else np.ones(n)
        order = np.argsort(-regret)

        new_assign = np.full(n, -1, dtype=int)
        used = np.zeros(k, dtype=int)
        for idx in order:
            for c in np.argsort(d[idx]):
                if used[c] < target[c]:
                    new_assign[idx] = int(c)
                    used[c] += 1
                    break

        new_centers = centers.copy()
        for c in range(k):
            members = pts[new_assign == c]
            if len(members):
                new_centers[c] = members.mean(axis=0)

        if np.array_equal(new_assign, assignment):
            assignment = new_assign
            break
        assignment = new_assign
        centers = new_centers

    return assignment


def postcode_coord(plz: str) -> Optional[Tuple[float, float]]:
    temp = geocode_postcodes((str(plz).zfill(5),))
    if temp.empty or temp[["latitude", "longitude"]].isna().any(axis=None):
        return None
    return float(temp.iloc[0]["latitude"]), float(temp.iloc[0]["longitude"])


def nearest_neighbor_order(df: pd.DataFrame, depot: Optional[Tuple[float, float]]) -> List[int]:
    if df.empty:
        return []
    coords = [(float(r.latitude), float(r.longitude)) for r in df.itertuples()]
    remaining = set(range(len(coords)))

    if depot is None:
        centroid = (float(df["latitude"].mean()), float(df["longitude"].mean()))
        start = min(remaining, key=lambda i: haversine_km(centroid, coords[i]))
    else:
        start = min(remaining, key=lambda i: haversine_km(depot, coords[i]))

    order = [start]
    remaining.remove(start)
    while remaining:
        cur = coords[order[-1]]
        nxt = min(remaining, key=lambda i: haversine_km(cur, coords[i]))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def path_distance(df: pd.DataFrame, depot: Optional[Tuple[float, float]], return_to_depot: bool) -> float:
    if df.empty:
        return 0.0
    coords = [(float(r.latitude), float(r.longitude)) for r in df.itertuples()]
    total = 0.0
    if depot:
        total += haversine_km(depot, coords[0])
    for a, b in zip(coords, coords[1:]):
        total += haversine_km(a, b)
    if depot and return_to_depot:
        total += haversine_km(coords[-1], depot)
    return total


def two_opt(df: pd.DataFrame, depot: Optional[Tuple[float, float]], return_to_depot: bool) -> pd.DataFrame:
    if len(df) < 4:
        return df.reset_index(drop=True)
    best = df.reset_index(drop=True).copy()
    best_dist = path_distance(best, depot, return_to_depot)
    improved = True
    rounds = 0
    while improved and rounds < 5:
        improved = False
        rounds += 1
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = pd.concat(
                    [best.iloc[:i], best.iloc[i:j][::-1], best.iloc[j:]],
                    ignore_index=True,
                )
                d = path_distance(candidate, depot, return_to_depot)
                if d + 0.01 < best_dist:
                    best = candidate
                    best_dist = d
                    improved = True
    return best


def route_order(df: pd.DataFrame, depot: Optional[Tuple[float, float]], return_to_depot: bool) -> pd.DataFrame:
    idx = nearest_neighbor_order(df, depot)
    ordered = df.iloc[idx].reset_index(drop=True)
    return two_opt(ordered, depot, return_to_depot)


# ------------------------------------------------------------
# Plan bauen / Drag-Drop
# ------------------------------------------------------------
def make_item_label(row: pd.Series) -> str:
    sap = int(row["SAP"]) if pd.notna(row["SAP"]) else "-"
    name = str(row["Name"]).strip()
    ort = str(row["Ort"]).strip()
    source = str(row["Quelle"])
    old = str(row.get("Bisherige_Touren", "")).strip()
    return f"{sap} · {ort} · {name} · {old} · [{source}]"


def build_plan(
    customers: pd.DataFrame,
    max_stops: int,
    separate_sources: bool,
    depot_plz: Dict[str, str],
    return_to_depot: bool,
) -> List[Dict[str, object]]:
    working = customers.dropna(subset=["latitude", "longitude"]).copy()
    containers: List[Dict[str, object]] = []
    route_counter = 1

    groups: Iterable[Tuple[str, pd.DataFrame]]
    if separate_sources:
        groups = working.groupby("Quelle", sort=False)
    else:
        groups = [("MIX", working)]

    for source, grp in groups:
        grp = grp.reset_index(drop=True)
        labels = balanced_kmeans(grp, max_stops)
        grp = grp.assign(_cluster=labels)
        dep = postcode_coord(depot_plz.get(source, "")) if source != "MIX" else None

        for cluster_id, route_df in grp.groupby("_cluster", sort=True):
            route_df = route_df.drop(columns="_cluster").copy()
            route_df = route_order(route_df, dep, return_to_depot)
            prefix = source.replace("HUPA_", "") if separate_sources else "FT"
            header = f"FT-{prefix}-{route_counter:02d}"
            items = [make_item_label(r) for _, r in route_df.iterrows()]
            containers.append({"header": header, "items": items})
            route_counter += 1

    missing_geo = customers[customers[["latitude", "longitude"]].isna().any(axis=1)]
    containers.append(
        {
            "header": "NICHT EINGEPLANT",
            "items": [make_item_label(r) for _, r in missing_geo.iterrows()],
        }
    )
    return containers


def plan_to_assignment(containers: List[Dict[str, object]], label_map: Dict[str, str]) -> Dict[str, Tuple[str, int]]:
    result: Dict[str, Tuple[str, int]] = {}
    for container in containers:
        header = str(container["header"])
        for pos, label in enumerate(container.get("items", []), start=1):
            uid = label_map.get(label)
            if uid:
                result[uid] = (header, pos)
    return result


def assignment_frame(
    source_df: pd.DataFrame,
    containers: List[Dict[str, object]],
    label_map: Dict[str, str],
) -> pd.DataFrame:
    ass = plan_to_assignment(containers, label_map)
    rows = []
    by_uid = source_df.set_index("uid", drop=False)
    for uid, (route, pos) in ass.items():
        if uid not in by_uid.index:
            continue
        r = by_uid.loc[uid]
        rows.append(
            {
                "Neue Tour": route,
                "Reihenfolge": pos,
                "Quelle": r["Quelle"],
                "CSB": r["CSB"],
                "SAP": r["SAP"],
                "Name": r["Name"],
                "Strasse": r["Strasse"],
                "PLZ": r["Plz"],
                "Ort": r["Ort"],
                "Bisherige Touren": r["Bisherige_Touren"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "uid": uid,
            }
        )
    return pd.DataFrame(rows)


def current_route_stats(plan_df: pd.DataFrame, depot_plz: Dict[str, str], return_to_depot: bool) -> pd.DataFrame:
    rows = []
    if plan_df.empty:
        return pd.DataFrame()
    for route, grp in plan_df[plan_df["Neue Tour"] != "NICHT EINGEPLANT"].groupby("Neue Tour", sort=False):
        grp = grp.sort_values("Reihenfolge")
        source = grp["Quelle"].iloc[0] if grp["Quelle"].nunique() == 1 else "MIX"
        depot = postcode_coord(depot_plz.get(source, "")) if source != "MIX" else None
        geo_km = path_distance(grp.rename(columns={"PLZ":"Plz"}), depot, return_to_depot)
        # Luftlinien-basierte Schätzung. Bewusst als Schätzung gekennzeichnet.
        est_road_km = geo_km * 1.25
        rows.append(
            {
                "Tour": route,
                "Kunden": len(grp),
                "Bereich": source,
                "Geo-km": round(geo_km, 1),
                "Straßen-km ~": round(est_road_km, 1),
            }
        )
    return pd.DataFrame(rows)


def optimize_containers(
    containers: List[Dict[str, object]],
    source_df: pd.DataFrame,
    label_map: Dict[str, str],
    depot_plz: Dict[str, str],
    return_to_depot: bool,
) -> List[Dict[str, object]]:
    uid_to_label = {uid: label for label, uid in label_map.items()}
    by_uid = source_df.set_index("uid", drop=False)
    out: List[Dict[str, object]] = []

    for c in containers:
        header = str(c["header"])
        labels = list(c.get("items", []))
        if header == "NICHT EINGEPLANT" or len(labels) < 2:
            out.append({"header": header, "items": labels})
            continue
        uids = [label_map[x] for x in labels if x in label_map and label_map[x] in by_uid.index]
        route = by_uid.loc[uids].copy()
        if isinstance(route, pd.Series):
            route = route.to_frame().T
        source = route["Quelle"].iloc[0] if route["Quelle"].nunique() == 1 else "MIX"
        depot = postcode_coord(depot_plz.get(source, "")) if source != "MIX" else None
        ordered = route_order(route.reset_index(drop=True), depot, return_to_depot)
        out.append({"header": header, "items": [uid_to_label[u] for u in ordered["uid"].tolist()]})
    return out


def plan_excel_bytes(plan_df: pd.DataFrame, planning_date: date, days: List[str]) -> bytes:
    buffer = io.BytesIO()
    export = plan_df.copy()
    export.insert(0, "Planungsdatum", planning_date.strftime("%d.%m.%Y"))
    export.insert(1, "Ausgangs-Liefertage", ", ".join(DAY_LABELS[d] for d in days))
    export = export.drop(columns=["latitude", "longitude", "uid"], errors="ignore")

    planned = export[export["Neue Tour"] != "NICHT EINGEPLANT"].copy()
    unplanned = export[export["Neue Tour"] == "NICHT EINGEPLANT"].copy()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        planned.to_excel(writer, sheet_name="Feiertagstouren", index=False)
        unplanned.to_excel(writer, sheet_name="Nicht eingeplant", index=False)
        wb = writer.book
        for sheet_name, frame in [("Feiertagstouren", planned), ("Nicht eingeplant", unplanned)]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            header_fmt = wb.add_format({"bold": True, "bg_color": "#332B3E", "font_color": "#FFFFFF", "border": 0})
            for col_idx, col in enumerate(frame.columns):
                ws.write(0, col_idx, col, header_fmt)
                max_len = max([len(str(col))] + [len(str(v)) for v in frame[col].head(300).fillna("")])
                ws.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 38))
    return buffer.getvalue()


def render_map(plan_df: pd.DataFrame):
    points = plan_df[(plan_df["Neue Tour"] != "NICHT EINGEPLANT") & plan_df["latitude"].notna()].copy()
    if points.empty:
        st.info("Keine geokodierten Kunden für die Karte vorhanden.")
        return

    route_names = list(points["Neue Tour"].drop_duplicates())
    color_map = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(route_names)}
    points["color"] = points["Neue Tour"].map(color_map)
    points["tooltip"] = points.apply(
        lambda r: f"{r['Neue Tour']} · {r['Reihenfolge']}\n{r['SAP']} · {r['Name']}\n{r['PLZ']} {r['Ort']}",
        axis=1,
    )

    paths = []
    for route, grp in points.sort_values(["Neue Tour", "Reihenfolge"]).groupby("Neue Tour", sort=False):
        path = [[float(x.longitude), float(x.latitude)] for x in grp.itertuples()]
        if len(path) >= 2:
            paths.append({"route": route, "path": path, "color": color_map[route]})

    layers = [
        pdk.Layer(
            "PathLayer",
            data=paths,
            get_path="path",
            get_color="color",
            width_min_pixels=3,
            opacity=0.75,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=3500,
            radius_min_pixels=5,
            radius_max_pixels=10,
            pickable=True,
        ),
    ]

    view = pdk.ViewState(
        latitude=float(points["latitude"].mean()),
        longitude=float(points["longitude"].mean()),
        zoom=6.0,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view,
            tooltip={"text": "{tooltip}"},
            map_style=None,
        ),
        use_container_width=True,
    )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧭 Feiertags-Tourenplaner")
st.caption("Die App baut einen Geo-Vorschlag. Danach kannst du Kunden frei zwischen Touren verschieben, die Reihenfolge ändern oder sie in ›Nicht eingeplant‹ ablegen.")

uploaded = st.file_uploader("Quelldatei hochladen", type=["xlsx", "xls"])

if uploaded is None:
    st.info("Bitte deine Quelldatei hochladen. Verwendet werden ausschließlich die ersten vier Bereiche: DIREKT, MK, HUPA_NMS und HUPA_MALCHOW.")
    st.stop()

try:
    raw = load_source(uploaded.getvalue())
except Exception as exc:
    st.error(f"Datei konnte nicht gelesen werden: {exc}")
    st.stop()

with st.sidebar:
    st.header("Planung")
    planning_date = st.date_input("Planungs-/Feiertagsdatum", value=date.today())
    days = st.multiselect(
        "Welche regulären Liefertage sollen in die Planung?",
        options=DAY_COLUMNS,
        default=["Fr"],
        format_func=lambda x: DAY_LABELS[x],
    )
    sources = st.multiselect("Bereiche", SHEETS, default=SHEETS)
    max_stops = st.slider("Max. Kunden je Tour", min_value=3, max_value=20, value=8, step=1)
    separate_sources = st.toggle("Bereiche getrennt planen", value=True)
    return_to_depot = st.toggle("Rückfahrt zum Startpunkt mitrechnen", value=True)

    st.divider()
    st.subheader("Startpunkt / Depot (PLZ)")
    depot_plz = {}
    for s in SHEETS:
        depot_plz[s] = st.text_input(s, value=DEFAULT_DEPOT_PLZ[s], max_chars=5, key=f"dep_{s}")

selected = filter_customers(raw, days, sources)
if selected.empty:
    st.warning("Für diese Auswahl wurden keine Kunden gefunden.")
    st.stop()

if pgeocode is None:
    st.error("Das Paket `pgeocode` fehlt. Bitte `pip install pgeocode` ausführen bzw. requirements.txt verwenden.")
    st.stop()

with st.spinner("PLZ-Koordinaten werden vorbereitet …"):
    selected = add_geo(selected)

selected["label"] = selected.apply(make_item_label, axis=1)
label_map = dict(zip(selected["label"], selected["uid"]))

valid_geo = selected[["latitude", "longitude"]].notna().all(axis=1).sum()
invalid_geo = len(selected) - valid_geo

m1, m2, m3, m4 = st.columns(4)
m1.metric("Kunden", len(selected))
m2.metric("Geo erkannt", int(valid_geo))
m3.metric("ohne Geo", int(invalid_geo))
if separate_sources:
    expected_tours = sum(
        math.ceil(n / max_stops)
        for n in selected[selected[["latitude", "longitude"]].notna().all(axis=1)].groupby("Quelle").size().tolist()
    )
else:
    expected_tours = max(1, math.ceil(valid_geo / max_stops)) if valid_geo else 0
m4.metric("voraussichtliche Touren", int(expected_tours))

st.markdown(
    "<span class='small-note'>Geo-Basis: PLZ-Mittelpunkt. Die Straßen-km in der Übersicht sind nur eine grobe Schätzung und noch keine echte LKW-Routenberechnung.</span>",
    unsafe_allow_html=True,
)

config_signature = (
    tuple(days), tuple(sources), int(max_stops), bool(separate_sources), bool(return_to_depot), tuple(sorted(depot_plz.items()))
)

col_a, col_b, col_c = st.columns([1, 1, 3])
with col_a:
    rebuild = st.button("⚡ Touren neu bauen", type="primary", use_container_width=True)
with col_b:
    optimize = st.button("↻ Reihenfolge optimieren", use_container_width=True)

if "plan_containers" not in st.session_state or rebuild:
    st.session_state.plan_containers = build_plan(
        selected,
        max_stops=max_stops,
        separate_sources=separate_sources,
        depot_plz=depot_plz,
        return_to_depot=return_to_depot,
    )
    st.session_state.plan_signature = config_signature
    st.session_state.sort_version = st.session_state.get("sort_version", 0) + 1

# Wenn Auswahl geändert wurde, nicht stillschweigend überschreiben.
if st.session_state.get("plan_signature") != config_signature:
    st.warning("Die Planungsparameter wurden geändert. Klicke auf **Touren neu bauen**, um den Vorschlag neu zu berechnen. Deine aktuelle manuelle Planung bleibt bis dahin erhalten.")

if optimize:
    st.session_state.plan_containers = optimize_containers(
        st.session_state.plan_containers,
        selected,
        label_map,
        depot_plz,
        return_to_depot,
    )
    st.session_state.sort_version = st.session_state.get("sort_version", 0) + 1

st.subheader("Touren per Drag & Drop bearbeiten")
st.caption("Kunden zwischen Touren ziehen · innerhalb einer Tour Reihenfolge ändern · zum Entfernen aus der Feiertagsplanung nach ›NICHT EINGEPLANT‹ ziehen.")

sortable_style = """
.sortable-component {font-size: 13px;}
.sortable-container {background: #17151d; border: 1px solid #3a3542; border-radius: 12px; padding: 8px; margin: 6px; min-width: 300px;}
.sortable-container-header {background: #2b2533; color: #f4f1f7; border-radius: 8px; padding: 8px 10px; font-weight: 700;}
.sortable-container-body {background: #17151d; min-height: 55px;}
.sortable-item {background: #27222e; color: #f2eef6; border: 1px solid #42394d; border-radius: 8px; padding: 7px 9px; margin: 5px 0; cursor: grab;}
.sortable-item:hover {background: #342c3e;}
"""

# Der Key wird nur bei bewusstem Neuaufbau/Optimieren geändert. Das umgeht
# bekannte Component-Probleme beim nachträglichen Ändern des Item-Sets.
sorted_result = sort_items(
    st.session_state.plan_containers,
    multi_containers=True,
    direction="vertical",
    custom_style=sortable_style,
    key=f"route_sort_{st.session_state.get('sort_version', 0)}",
)
if sorted_result:
    st.session_state.plan_containers = sorted_result

plan_df = assignment_frame(selected, st.session_state.plan_containers, label_map)

st.divider()
st.subheader("Aktuelle Planung")

stats = current_route_stats(plan_df, depot_plz, return_to_depot)
if not stats.empty:
    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Geo-km": st.column_config.NumberColumn(format="%.1f km"),
            "Straßen-km ~": st.column_config.NumberColumn(format="%.1f km"),
        },
    )

planned_count = int((plan_df["Neue Tour"] != "NICHT EINGEPLANT").sum()) if not plan_df.empty else 0
unplanned_count = int((plan_df["Neue Tour"] == "NICHT EINGEPLANT").sum()) if not plan_df.empty else 0
st.caption(f"{planned_count} Kunden eingeplant · {unplanned_count} nicht eingeplant")

map_tab, table_tab, export_tab = st.tabs(["🗺️ Karte", "📋 Kundenliste", "⬇️ Export"])
with map_tab:
    render_map(plan_df)

with table_tab:
    show = plan_df.drop(columns=["latitude", "longitude", "uid"], errors="ignore")
    st.dataframe(show, use_container_width=True, hide_index=True, height=650)

with export_tab:
    export_bytes = plan_excel_bytes(plan_df, planning_date, days)
    st.download_button(
        "Excel mit aktueller Planung herunterladen",
        data=export_bytes,
        file_name=f"Feiertagstouren_{planning_date:%Y-%m-%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("Der Export enthält die manuell verschobene Reihenfolge und ein eigenes Blatt für nicht eingeplante Kunden.")
